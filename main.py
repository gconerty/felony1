from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime, timedelta
import re # For parsing the category from LLM output and weapon keywords
import google.generativeai as genai
import time
import calendar # Needed for finding the last day of the month

# Initialize Flask application
app = Flask(__name__)

# --- Google Gemini API Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_model = None

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not found. LLM calls will fail.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"Error configuring or initializing Google Gemini Model: {e}")
        gemini_model = None

# --- Define "No Go" Categories (used in prompt and policy check) ---
NO_GO_CATEGORIES = [
    "Violent Crime Involving a Weapon",
    "Sexual Related Crime",
    "Murder or Manslaughter",
    "Distribution Drug Related Crime",
    "Human Trafficking"
]

# --- Define Critical Keywords for Input Scan (Safety Net) ---
# These keywords in the *original input* might trigger a denial even if LLM says "Other".
# Be careful not to make these too broad.
CRITICAL_INPUT_KEYWORDS = {
    # Category: Keywords
    "Violent Crime Involving a Weapon": ["armed", "weapon", "firearm", "gun", "knife", "stabbing", "shooting"],
    "Sexual Related Crime": ["sexual assault", "rape", "molestation", "child pornography", "non-consensual sex", "involuntary sex"],
    "Murder or Manslaughter": ["murder", "manslaughter", "homicide"],
    "Distribution Drug Related Crime": ["sell drugs", "distribute drugs", "manufacture drugs", "transport drugs", "traffic drugs", "deliver drugs", "cultivate drugs", "intent to distribute", "intent to sell"],
    "Human Trafficking": ["human trafficking", "forced labor", "sexual exploitation trafficking"]
    # Add "assault", "battery" here ONLY if *any* felony assault/battery (even without weapon) is a No Go.
    # If simple assault/battery is okay, leave them out of this specific list.
    # "Violent Crime Without Weapon": ["assault", "battery"] # Example if simple assault is also No Go
}


# --- Weapon Keyword Detection ---
WEAPON_KEYWORDS = [
    "weapon", "gun", "firearm", "handgun", "shotgun", "rifle", "pistol",
    "knife", "dagger", "blade", "stabbing", "slashing",
    "armed", "explosive", "bomb", "grenade", "brass knuckles",
    "club", "bat", "blunt object", "crowbar", "tire iron", "deadly weapon",
    "assault rifle", "machine gun", "shooter", "shooting"
]

def text_mentions_weapon(description):
    """
    Scans the description text for weapon-related keywords.
    """
    if not description:
        return False
    description_lower = description.lower()
    for keyword in WEAPON_KEYWORDS:
        if keyword in ["armed", "bat", "gun", "knife", "club"]:
            if re.search(r'\b' + re.escape(keyword) + r'\b', description_lower):
                return True
        elif keyword in description_lower:
            return True
    return False

# --- Flexible Date Parsing Helper ---
def parse_flexible_date(date_str):
    """
    Attempts to parse a date string in YYYY-MM-DD, YYYY-MM, or YYYY formats.
    """
    date_str = date_str.strip()
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            dt = datetime.strptime(date_str, '%Y-%m')
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            return dt.replace(day=last_day)
        except ValueError:
            try:
                dt = datetime.strptime(date_str, '%Y')
                return dt.replace(month=12, day=31)
            except ValueError:
                return None

# --- LLM Interaction with Categorization ---
# (query_gemini_and_categorize function remains the same as the previous version)
def query_gemini_and_categorize(felony_description, weapon_checked_by_user=None, weapon_found_in_text=False, max_retries=3, retry_delay=5):
    """
    Queries Google Gemini to get a legal summary and explicit categorization of a felony.
    """
    if not GOOGLE_API_KEY:
        return "Error", "", "Google API key not configured."
    if not gemini_model:
        return "Error", "", "Gemini model not initialized."

    category_list_for_prompt = "\n".join([f"- {cat}" for cat in NO_GO_CATEGORIES])

    weapon_guidance_parts = []
    if weapon_checked_by_user is True:
        weapon_guidance_parts.append("User checkbox indicates a weapon WAS involved.")
    elif weapon_checked_by_user is False:
        weapon_guidance_parts.append("User checkbox indicates a weapon was NOT involved.")
    else:
        weapon_guidance_parts.append("User did not specify weapon involvement via checkbox.")

    if weapon_found_in_text:
        weapon_guidance_parts.append("Text analysis of the description suggests a weapon was mentioned.")
    else:
        weapon_guidance_parts.append("Text analysis of the description did not find common weapon keywords.")

    combined_weapon_guidance = " ".join(weapon_guidance_parts)

    prompt = f"""
You are a legal assistant. Analyze the following felony description.
Your primary task is to categorize the offense based *only* on the information explicitly provided in the description. Do not infer additional charges or circumstances not stated.

Guidance on Weapon Involvement:
{combined_weapon_guidance}

Instructions:
1.  First, determine the most fitting category for the described offense from the list below.
    Consider both the felony description and the 'Guidance on Weapon Involvement' provided above.
    - If the overall evidence (description, checkbox, text scan) strongly suggests a violent crime AND a weapon, "Violent Crime Involving a Weapon" is a strong candidate.
    - If the overall evidence strongly suggests NO weapon was involved (e.g., checkbox says no AND text scan is negative, OR description is explicit), DO NOT use "Violent Crime Involving a Weapon". In such cases, if the act is still a prohibited category (e.g. Murder), choose that. Otherwise, choose "Other".
    - For drug-related offenses:
        - If the description explicitly mentions selling, transporting, delivering, cultivating, manufacturing, trafficking, or intent to distribute/sell, categorize it as "Distribution Drug Related Crime".
        - If the description *only* mentions possession of drugs, even if a felony, categorize it as "Other" UNLESS it also clearly falls into another prohibited category from the list. Do not infer distribution from possession alone.
    If it fits multiple categories from the list, choose the most severe or prominent one that accurately reflects all provided information.
    If it does not clearly fit any of these specific categories from the list, categorize it as "Other".
    Output this category on a single, separate line formatted EXACTLY as:
    Primary Category: [Chosen Category Name or Other]

2.  After the "Primary Category:" line, provide a concise legal explanation of the alleged crime under U.S. law. Focus on the key elements. Do not make assumptions about guilt or innocence.

Categories to choose from:
{category_list_for_prompt}
- Other

Felony Description to analyze: "{felony_description}"
    """

    generation_config = genai.types.GenerationConfig(
        temperature=0.15,
        max_output_tokens=750
    )

    llm_full_output = ""
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )

            if response.parts:
                llm_full_output = response.text.strip()
                category_match = re.search(r"Primary Category:\s*(.+)", llm_full_output, re.IGNORECASE)
                if category_match:
                    parsed_category = category_match.group(1).strip()
                    for known_cat in NO_GO_CATEGORIES:
                        if known_cat.lower() == parsed_category.lower():
                            return known_cat, llm_full_output, None
                    if parsed_category.lower() == "other":
                         return "Other", llm_full_output, None
                    print(f"Warning: LLM returned category '{parsed_category}' not in predefined list or 'Other'. Treating as 'Other'. Full output: {llm_full_output[:200]}")
                    return "Other", llm_full_output, None
                else:
                    error_msg = "Error: LLM did not provide category in expected format."
                    print(f"Gemini API attempt {attempt + 1}: {error_msg} Full output: {llm_full_output[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return "Error", llm_full_output, error_msg
            else: # No parts in response
                error_details = []
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_details.append(f"Prompt blocked due to: {response.prompt_feedback.block_reason.name}")
                if response.candidates:
                    for cand in response.candidates:
                        if cand.finish_reason and cand.finish_reason.name != 'STOP':
                            error_details.append(f"Generation finished due to: {cand.finish_reason.name}")
                        if cand.safety_ratings:
                            problematic_ratings = [(r.category.name, r.probability.name) for r in cand.safety_ratings if r.probability.name not in ['NEGLIGIBLE', 'LOW']]
                            if problematic_ratings: error_details.append(f"Safety Ratings Issues: {problematic_ratings}")
                error_msg = "Error: No content generated by LLM."
                if error_details: error_msg += " Details: " + "; ".join(error_details)
                print(f"Gemini API attempt {attempt + 1} returned no valid content: {error_msg}")
                if any(block_reason in error_msg.upper() for block_reason in ["SAFETY", "BLOCK_REASON_OTHER"]):
                    return "Error", "", error_msg
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return "Error", "", error_msg

        except Exception as e:
            if "429" in str(e) and "quota" in str(e).lower():
                print(f"Quota exceeded on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    match_retry = re.search(r"retry_delay\s*\{\s*seconds\s*:\s*(\d+)\s*\}", str(e))
                    dynamic_retry_delay = int(match_retry.group(1)) if match_retry else retry_delay + (attempt * 5)
                    print(f"Retrying in {dynamic_retry_delay} seconds...")
                    time.sleep(dynamic_retry_delay)
                    continue
                else:
                    return "Error", "", f"LLM Quota Error after {max_retries} attempts: {e}"
            print(f"General error querying Gemini (attempt {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
            if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e):
                return "Error", "", f"LLM Auth/Permission Error: {e}"
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "Error", "", f"LLM Error after {max_retries} attempts: {e}"
    return "Error", llm_full_output, "Max retries exceeded for LLM API request."


# --- Policy Violation Check ---
def check_policy_violation(llm_category, felony_dates_str_list=None, release_date_str=None):
    """
    Checks LLM category and dates. Returns (is_approved, reason).
    """
    # Check 1: LLM Category
    if llm_category in NO_GO_CATEGORIES:
        return False, f"Denied based on LLM category: '{llm_category}'"
    if llm_category == "Error":
        return False, "Denied due to error in LLM processing or categorization."

    # Check 2: Date-related checks
    error_message = "Invalid date format. Please use YYYY-MM-DD, YYYY-MM, or YYYY."
    felony_datetimes = []
    if felony_dates_str_list:
        for date_str in felony_dates_str_list:
            parsed_date = parse_flexible_date(date_str)
            if parsed_date:
                felony_datetimes.append(parsed_date)
            elif date_str:
                return False, f"{error_message} (Offending value: '{date_str}')"

    today = datetime.today()

    # Rule: Felony within a year
    if felony_datetimes:
        for date_obj in felony_datetimes:
            if today - date_obj <= timedelta(days=365):
                return False, "Denied: Felony conviction within the last year."

    # Rule: Released from Jail within a year
    release_date = None
    if release_date_str and release_date_str.strip():
        release_date = parse_flexible_date(release_date_str)
        if not release_date:
            return False, f"{error_message} (Offending value: '{release_date_str}')"
        if today - release_date <= timedelta(days=365):
            return False, "Denied: Released from jail within the last year."

    # Rule: 2 or more felonies unless the most recent felony is more than 10 years old.
    if felony_datetimes and len(felony_datetimes) >= 2:
        most_recent_felony_date = max(felony_datetimes)
        if (today - most_recent_felony_date).days <= (10 * 365.25):
            return False, "Denied: Two or more felony convictions, and the most recent is within the last 10 years."

    return True, "Approved: No policy violations based on LLM category and date checks."


# --- Helper for Critical Keyword Check ---
def check_input_for_critical_keywords(felony_input):
    """
    Scans the original input for critical keywords associated with No Go categories.
    Returns the category name if a keyword is found, otherwise None.
    """
    if not felony_input:
        return None
    input_lower = felony_input.lower()
    for category, keywords in CRITICAL_INPUT_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary for potentially ambiguous keywords
            if keyword in ["armed", "gun", "knife", "bat", "club", "murder"]:
                 if re.search(r'\b' + re.escape(keyword) + r'\b', input_lower):
                    return category # Return the category name
            elif keyword in input_lower:
                return category # Return the category name
    return None


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index_route():
    result_status = None
    reason_message = ""
    llm_full_text = ""
    llm_assigned_category = ""
    weapon_involved_checked_on_form = False
    weapon_detected_in_text_on_form = False
    final_decision_cat_display = "" # For displaying the category used in final decision

    if request.method == 'POST':
        felony_input = request.form.get('felony', '')
        felony_dates_input = request.form.get('felony_dates', '')
        release_date_input = request.form.get('release_date', '')

        weapon_checkbox_val = request.form.get('weapon_involved')
        weapon_checked_by_user = True if weapon_checkbox_val == 'on' else False
        weapon_involved_checked_on_form = weapon_checked_by_user

        weapon_found_in_text = text_mentions_weapon(felony_input)
        weapon_detected_in_text_on_form = weapon_found_in_text

        felony_dates_str_list = []
        if felony_dates_input:
            felony_dates_str_list = [date.strip() for date in felony_dates_input.split(',') if date.strip()]

        if not felony_input:
            result_status = "Error"
            reason_message = "Felony description cannot be empty."
        else:
            category, full_output, error_msg_from_llm = query_gemini_and_categorize(
                felony_input,
                weapon_checked_by_user=weapon_checked_by_user,
                weapon_found_in_text=weapon_found_in_text
            )
            llm_full_text = full_output
            llm_assigned_category = category # Store the original LLM category

            final_decision_category = category # Start with LLM's decision

            # Apply Overrides
            if category == "Violent Crime Involving a Weapon" and not weapon_checked_by_user and not weapon_found_in_text:
                print(f"Overriding LLM category. Original: {category}. Checkbox: No, Text Scan: No. New Category: Other")
                final_decision_category = "Other"
            elif category == "Other" and (weapon_checked_by_user or weapon_found_in_text):
                 print(f"Potential Override: LLM said 'Other', but weapon indicated. Considering as 'Violent Crime Involving a Weapon'.")
                 final_decision_category = "Violent Crime Involving a Weapon"
            elif category == "Distribution Drug Related Crime":
                description_lower = felony_input.lower()
                is_just_possession = "possession" in description_lower and not any(dist_kw in description_lower for dist_kw in ["sell", "distribute", "manufacture", "transport", "traffic", "deliver", "cultivate", "intent to"])
                if is_just_possession:
                    print(f"Overriding LLM category. Original: {category}, Input appears possession only. New Category: Other")
                    final_decision_category = "Other"

            final_decision_cat_display = final_decision_category # Store for display

            if error_msg_from_llm or final_decision_category == "Error":
                result_status = "Error"
                reason_message = error_msg_from_llm or "LLM failed to categorize or an error occurred."
            else:
                is_approved, policy_reason = check_policy_violation(
                    final_decision_category,
                    felony_dates_str_list,
                    release_date_input
                )

                # --- Input Keyword Safety Net Check ---
                if is_approved: # Only check keywords if it wasn't already denied
                    critical_keyword_category = check_input_for_critical_keywords(felony_input)
                    if critical_keyword_category:
                         # Check if the category found by keyword is different from the final decision category
                         # This prevents denying twice for the same reason if LLM already caught it.
                         if final_decision_category != critical_keyword_category:
                            print(f"Keyword Safety Net Triggered: Found '{critical_keyword_category}' related keyword in input '{felony_input}'. Overriding Approval.")
                            is_approved = False
                            policy_reason = f"Denied based on keyword check of input description (found term related to '{critical_keyword_category}')."

                result_status = "Approved" if is_approved else "Denied"
                reason_message = policy_reason

    felony_description_val = request.form.get('felony', '') if request.method == 'POST' else ''
    felony_dates_val = request.form.get('felony_dates', '') if request.method == 'POST' else ''
    release_date_val = request.form.get('release_date', '') if request.method == 'POST' else ''

    return render_template('index.html',
                           result=result_status,
                           reason=reason_message,
                           llm_output=llm_full_text,
                           llm_category=llm_assigned_category, # Show original LLM category
                           final_category=final_decision_cat_display, # Show category used for decision
                           felony_description_val=felony_description_val,
                           felony_dates_val=felony_dates_val,
                           release_date_val=release_date_val,
                           weapon_involved_val=weapon_involved_checked_on_form,
                           weapon_in_text_val=weapon_detected_in_text_on_form)


@app.route('/check_felony', methods=['POST'])
def check_felony_api():
    felony_input = request.form.get('felony','')
    felony_dates_input = request.form.get('felony_dates','')
    release_date_input = request.form.get('release_date','')
    weapon_checkbox_val = request.form.get('weapon_involved')
    weapon_checked_by_user = True if weapon_checkbox_val == 'on' else False

    weapon_found_in_text = text_mentions_weapon(felony_input)

    felony_dates_str_list = []
    if felony_dates_input:
        felony_dates_str_list = [date.strip() for date in felony_dates_input.split(',') if date.strip()]

    if not felony_input:
        return jsonify({"status": "Error", "reason": "No felony description provided.", "llm_category": "", "llm_full_output": ""})

    category, full_output, error_msg_from_llm = query_gemini_and_categorize(
        felony_input,
        weapon_checked_by_user=weapon_checked_by_user,
        weapon_found_in_text=weapon_found_in_text
    )
    llm_assigned_category_for_api = category

    final_decision_category = category

    # Apply Overrides
    if category == "Violent Crime Involving a Weapon" and not weapon_checked_by_user and not weapon_found_in_text:
        print(f"API: Overriding LLM category. Original: {category}. Checkbox: No, Text Scan: No. New Category: Other")
        final_decision_category = "Other"
    elif category == "Other" and (weapon_checked_by_user or weapon_found_in_text):
        print(f"API Notice: LLM said 'Other', but weapon indicated. Considering as 'Violent Crime Involving a Weapon'.")
        final_decision_category = "Violent Crime Involving a Weapon"
    elif category == "Distribution Drug Related Crime":
        description_lower = felony_input.lower()
        is_just_possession = "possession" in description_lower and not any(dist_kw in description_lower for dist_kw in ["sell", "distribute", "manufacture", "transport", "traffic", "deliver", "cultivate", "intent to"])
        if is_just_possession:
            print(f"API: Overriding LLM category. Original: {category}, Input appears possession only. New Category: Other")
            final_decision_category = "Other"

    if error_msg_from_llm or final_decision_category == "Error":
        return jsonify({"status": "Error", "reason": error_msg_from_llm or "LLM failed to categorize or an error occurred.",
                        "llm_category_by_llm": llm_assigned_category_for_api,
                        "final_category_used_for_decision": final_decision_category,
                        "weapon_checkbox_status": weapon_checked_by_user,
                        "weapon_detected_in_text": weapon_found_in_text,
                        "llm_full_output": full_output})
    else:
        is_approved, policy_reason = check_policy_violation(
            final_decision_category,
            felony_dates_str_list,
            release_date_input
        )

        # --- Input Keyword Safety Net Check for API ---
        final_status = "Approved" if is_approved else "Denied"
        final_reason = policy_reason
        keyword_override_reason = None

        if is_approved: # Only check keywords if not already denied by category/date
            critical_keyword_category = check_input_for_critical_keywords(felony_input)
            if critical_keyword_category:
                 if final_decision_category != critical_keyword_category:
                    print(f"API Keyword Safety Net Triggered: Found '{critical_keyword_category}' related keyword in input '{felony_input}'. Overriding Approval.")
                    final_status = "Denied"
                    is_approved = False # Ensure boolean matches
                    keyword_override_reason = f"Denied based on keyword check of input description (found term related to '{critical_keyword_category}')."
                    final_reason = keyword_override_reason # Use keyword reason if it overrides

        return jsonify({
            "status": final_status,
            "is_approved": is_approved,
            "reason": final_reason,
            "llm_category_by_llm": llm_assigned_category_for_api,
            "final_category_used_for_decision": final_decision_category,
            "keyword_override_reason": keyword_override_reason, # Indicate if keyword override happened
            "weapon_checkbox_status": weapon_checked_by_user,
            "weapon_detected_in_text": weapon_found_in_text,
            "llm_full_output": full_output
        })

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
