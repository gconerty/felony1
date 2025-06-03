from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime, timedelta
import re # For parsing the category from LLM output, weapon keywords, and date extraction
import google.generativeai as genai
import time
import calendar # Needed for finding the last day of the month

# Initialize Flask application
app = Flask(__name__)

# --- In-memory Log Storage ---
application_log = [] 
MAX_LOG_ENTRIES = 100 

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
CRITICAL_INPUT_KEYWORDS = {
    "Violent Crime Involving a Weapon": [
        "armed robbery", "assault with a deadly weapon", 
        "stabbing with", "shooting at", "shot someone", "fired a gun at", 
        "brandished a weapon during", 
        "used a gun in a robbery", "used a knife to threaten",
        "explosive device", "bombing", 
    ],
    "Sexual Related Crime": ["sexual assault", "rape", "molestation", "child pornography", "non-consensual sex", "involuntary sex", "sexual battery"],
    "Murder or Manslaughter": ["murder", "manslaughter", "homicide", "killed someone"],
    "Distribution Drug Related Crime": ["sell drugs", "distribute drugs", "manufacture drugs", "transport drugs", "traffic drugs", "deliver drugs", "cultivate drugs", "intent to distribute", "intent to sell"],
    "Human Trafficking": [
        "human trafficking", "trafficking of persons", "trafficking a person", "trafficking an individual",
        "forced labor", "involuntary servitude", "debt bondage", "compelled service",
        "sexual exploitation trafficking", "sex trafficking", "commercial sexual exploitation of a minor",
        "exploiting a person for labor", "exploiting an individual for sex", "coercing a person into labor", "coercing a person into prostitution",
        "transporting persons for exploitation", "harboring persons for exploitation",
        "recruiting persons for exploitation", "child trafficking", "minor trafficking", "trafficking of children",
        "person for prostitution", "individual for prostitution", "child for prostitution",
        "person smuggling", "people smuggling", "smuggling persons", "smuggling people", 
        "smuggling individuals", "smuggling children", "smuggling minors", "alien smuggling",
        "abduction of a child", "child abduction", "kidnapping a child", "kidnapping of a minor" 
    ]
}

# --- Weapon Keyword Detection ---
WEAPON_KEYWORDS = [
    "weapon", "gun", "firearm", "handgun", "shotgun", "rifle", "pistol", "revolver",
    "knife", "dagger", "blade", "stabbing", "slashing", "sword", "machete", "axe", "hatchet",
    "armed", "explosive", "bomb", "grenade", "missile", "detonator",
    "brass knuckles", "knuckles", "club", "bat", "baseball bat", "blunt object", "crowbar", 
    "tire iron", "pipe", "metal pipe", "stick", "heavy stick", "rock", "brick", "heavy object", "bottle", "glass bottle",
    "shovel", 
    "deadly weapon", "dangerous instrument", "improvised weapon",
    "assault rifle", "machine gun", "shooter", "shooting", "shot", "fired a"
]

# --- Violence Keywords (for distinguishing possession from violent use) ---
VIOLENCE_KEYWORDS = [
    "assault", "battery", "attack", "fight", "robbery", "threaten", "menace", "brandish", "beat", "beating", 
    "injure", "harm", "homicide", "murder", "manslaughter", "violent", "struck", "stabbed", "shot at", "used to harm", "used to threaten",
    "killed", "wounded", "victimized", "abduction", "kidnap", "kidnapping" 
]

def text_mentions_weapon(description):
    if not description: return False
    description_lower = description.lower()
    for keyword in WEAPON_KEYWORDS:
        # Use word boundaries for ALL keywords to prevent partial matches like "weapon" in "no weapon"
        if re.search(r'\b' + re.escape(keyword) + r'\b', description_lower):
            return True
    return False

def text_mentions_violence(description):
    """Checks if the description contains keywords indicative of a violent act."""
    if not description: return False
    description_lower = description.lower()
    for keyword in VIOLENCE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', description_lower):
            return True
    return False

# --- Super Flexible Date Parsing Helper ---
def parse_flexible_date(date_str):
    date_str = date_str.strip()
    month_replacements = {
        r'\bjanuary\b': 'Jan', r'\bfebruary\b': 'Feb', r'\bmarch\b': 'Mar',
        r'\bapril\b': 'Apr', r'\bmay\b': 'May', r'\bjune\b': 'Jun',
        r'\bjuly\b': 'Jul', r'\baugust\b': 'Aug', r'\bseptember\b': 'Sep',
        r'\boctober\b': 'Oct', r'\bnovember\b': 'Nov', r'\bdecember\b': 'Dec',
        r'\bjan\b': 'Jan', r'\bfeb\b': 'Feb', r'\bmar\b': 'Mar',
        r'\bapr\b': 'Apr', r'\bjun\b': 'Jun',
        r'\bjul\b': 'Jul', r'\baug\b': 'Aug', r'\bsep(t)?\b': 'Sep',
        r'\boct\b': 'Oct', r'\bnov\b': 'Nov', r'\bdec\b': 'Dec'
    }
    temp_date_str = date_str.lower()
    for pattern, replacement in month_replacements.items():
        temp_date_str = re.sub(pattern, replacement, temp_date_str)

    formats_to_try = [
        ('%Y-%m-%d', False), ('%m/%d/%Y', False), ('%m-%d-%Y', False),
        ('%b %Y', True), ('%B %Y', True), ('%b, %Y', True), ('%B, %Y', True),
        ('%Y-%m', True), ('%m/%Y', True), ('%m-%Y', True), ('%Y', True)
    ]
    for fmt, is_partial in formats_to_try:
        try:
            string_to_parse = temp_date_str if '%b' in fmt.lower() or '%B' in fmt.lower() else date_str
            dt = datetime.strptime(string_to_parse, fmt)
            if is_partial:
                if fmt in ['%Y-%m', '%m/%Y', '%m-%Y', '%b %Y', '%B %Y', '%b, %Y', '%B, %Y']: 
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    return dt.replace(day=last_day)
                elif fmt == '%Y': return dt.replace(month=12, day=31)
            return dt 
        except ValueError: continue 
    return None 

# --- Date Extraction from Text ---
def extract_dates_from_text(text_content):
    if not text_content: return []
    found_date_strings = []
    date_patterns = [
        r'\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b', 
        r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b', 
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4})\b',
        r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
        r'\b(\d{4}[-/]\d{1,2})\b',           
        r'\b(\d{1,2}[/-]\d{4})\b',           
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
        r'\b([12]\d{3})\b'                   
    ]
    temp_text = text_content
    def replace_and_collect(match_obj, collection):
        collection.append(match_obj.group(0)); return " " * len(match_obj.group(0))
    for pattern in date_patterns:
        flags = re.IGNORECASE if "jan" in pattern.lower() else 0
        temp_text = re.sub(pattern, lambda m: replace_and_collect(m, found_date_strings), temp_text, flags=flags)
    return list(set(found_date_strings))

# --- LLM Interaction with Multi-Offense Categorization ---
def query_gemini_for_offense_analysis(felony_description, weapon_checked_by_user=None, weapon_found_in_text=False, max_retries=3, retry_delay=5):
    if not GOOGLE_API_KEY: return ["Error"], "", "Google API key not configured."
    if not gemini_model: return ["Error"], "", "Gemini model not initialized."
    category_list_for_prompt = "\n".join([f"- {cat}" for cat in NO_GO_CATEGORIES])
    weapon_guidance_parts = []
    if weapon_checked_by_user is True: weapon_guidance_parts.append("User checkbox indicates a weapon WAS involved.")
    elif weapon_checked_by_user is False: weapon_guidance_parts.append("User checkbox indicates a weapon was NOT involved.")
    else: weapon_guidance_parts.append("User did not specify weapon involvement via checkbox.")
    if weapon_found_in_text: weapon_guidance_parts.append("Text analysis of the description suggests a weapon was mentioned.")
    else: weapon_guidance_parts.append("Text analysis of the description did not find common weapon keywords.")

    weapon_guidance_parts.append('Note: A "weapon" can include traditional arms as well as any object used to inflict or threaten serious harm (e.g., a stick, rock, heavy object, bottle, vehicle used intentionally to harm, shovel) if the context implies its use as such in a violent crime.')
    combined_weapon_guidance = " ".join(weapon_guidance_parts)

    prompt = f"""
You are a legal assistant. Analyze the following felony description, which may describe one or more distinct offenses.
Your primary task is to identify each distinct potential felony mentioned and categorize each one based *only* on the information explicitly provided in the description. Do not infer additional charges or circumstances not stated.

Guidance on Weapon Involvement (applies to each offense if relevant):
{combined_weapon_guidance}

Instructions:
1.  Read the entire "Felony Description to analyze".
2.  Identify each distinct potential felony described.
3.  For EACH distinct offense identified, assign it the most fitting category from the "Categories to choose from" list below.
    - Consider the weapon guidance for each offense. Remember the broad definition of a weapon if used to cause harm.
    - For "Violent Crime Involving a Weapon": The description must indicate BOTH a violent act (e.g., assault, battery, robbery, threat with a weapon, beating) AND the involvement of a weapon (including improvised items like a shovel, stick, rock, used to harm) in that specific violent act. Mere possession of a weapon (e.g., "felony possession of a firearm"), even if a felony, without a described violent act *using* that weapon, should be categorized as "Other" unless it fits another prohibited category.
    - For drug-related offenses: If it's only possession, categorize as "Other". If it's selling, manufacturing, trafficking, etc., use "Distribution Drug Related Crime".
    - For "Human Trafficking": Consider descriptions involving the recruitment, transportation, harboring, or receipt of persons (adults, children, individuals) through force, fraud, or coercion for the purpose of exploitation (like forced labor or sexual exploitation), OR descriptions involving the illegal smuggling of persons (e.g., "person smuggling," "alien smuggling"), OR offenses like "child abduction" or "kidnapping of a child". Even if terms like "accidental" are used by the input for smuggling/abduction, if the act itself is a felony, it should be considered for this category.
    - If an offense doesn't fit a specific prohibited category, assign "Other".
4.  Format your output for the categories by listing each identified offense and its category. Start each category on a new line, prefixed with "Identified Category: ".
    Example of format for multiple offenses:
    Identified Category: [Category for first offense or Other]
    Identified Category: [Category for second offense or Other]
    (If only one offense, list one "Identified Category:" line. If no specific offenses classifiable into the list are found, output one line: "Identified Category: Other")

5.  After listing all "Identified Category:" lines, provide a single, concise overall legal explanation of the alleged crimes under U.S. law.

Categories to choose from:
{category_list_for_prompt}
- Other
Felony Description to analyze: "{felony_description}"
    """
    generation_config = genai.types.GenerationConfig(temperature=0.15, max_output_tokens=1000)
    llm_full_output = ""
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt, generation_config=generation_config)
            if response.parts:
                llm_full_output = response.text.strip()
                identified_categories = re.findall(r"Identified Category:\s*(.+)", llm_full_output, re.IGNORECASE)
                if not identified_categories and ("Identified Category: Other" in llm_full_output or "Primary Category: Other" in llm_full_output):
                     return ["Other"], llm_full_output, None 
                normalized_categories = []
                if identified_categories:
                    for cat_str in identified_categories:
                        cat_str_clean = cat_str.strip()
                        found_known = False
                        for known_cat in NO_GO_CATEGORIES:
                            if known_cat.lower() == cat_str_clean.lower():
                                normalized_categories.append(known_cat); found_known = True; break
                        if not found_known and cat_str_clean.lower() == "other": normalized_categories.append("Other")
                        elif not found_known:
                            print(f"Warning: LLM returned category '{cat_str_clean}' not in predefined list or 'Other'. Treating as 'Other'.")
                            normalized_categories.append("Other")
                    return normalized_categories if normalized_categories else ["Other"], llm_full_output, None 
                else: 
                    primary_category_match = re.search(r"Primary Category:\s*(.+)", llm_full_output, re.IGNORECASE) 
                    if primary_category_match:
                        parsed_category = primary_category_match.group(1).strip()
                        for known_cat in NO_GO_CATEGORIES:
                            if known_cat.lower() == parsed_category.lower(): return [known_cat], llm_full_output, None
                        if parsed_category.lower() == "other": return ["Other"], llm_full_output, None
                        print(f"Warning: LLM used old 'Primary Category' format. Category '{parsed_category}' not in predefined list or 'Other'. Treating as ['Other'].")
                        return ["Other"], llm_full_output, None
                    error_msg = "Error: LLM did not provide categories in the expected 'Identified Category:' format."
                    print(f"Gemini API attempt {attempt + 1}: {error_msg} Full output: {llm_full_output[:500]}")
                    if attempt < max_retries - 1: time.sleep(retry_delay); continue
                    return ["Error"], llm_full_output, error_msg
            else: 
                error_details = []
                if response.prompt_feedback and response.prompt_feedback.block_reason: error_details.append(f"Prompt blocked due to: {response.prompt_feedback.block_reason.name}")
                if response.candidates:
                    for cand in response.candidates:
                        if cand.finish_reason and cand.finish_reason.name != 'STOP': error_details.append(f"Generation finished due to: {cand.finish_reason.name}")
                        if cand.safety_ratings:
                            problematic_ratings = [(r.category.name, r.probability.name) for r in cand.safety_ratings if r.probability.name not in ['NEGLIGIBLE', 'LOW']]
                            if problematic_ratings: error_details.append(f"Safety Ratings Issues: {problematic_ratings}")
                error_msg = "Error: No content generated by LLM."
                if error_details: error_msg += " Details: " + "; ".join(error_details)
                print(f"Gemini API attempt {attempt + 1} returned no valid content: {error_msg}")
                if any(block_reason in error_msg.upper() for block_reason in ["SAFETY", "BLOCK_REASON_OTHER"]): return ["Error"], "", error_msg 
                if attempt < max_retries - 1: time.sleep(retry_delay); continue
                return ["Error"], "", error_msg
        except Exception as e:
            if "429" in str(e) and "quota" in str(e).lower():
                print(f"Quota exceeded on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    match_retry = re.search(r"retry_delay\s*\{\s*seconds\s*:\s*(\d+)\s*\}", str(e))
                    dynamic_retry_delay = int(match_retry.group(1)) if match_retry else retry_delay + (attempt * 5)
                    print(f"Retrying in {dynamic_retry_delay} seconds...")
                    time.sleep(dynamic_retry_delay); continue
                else: return ["Error"], "", f"LLM Quota Error after {max_retries} attempts: {e}"
            print(f"General error querying Gemini (attempt {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
            if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e): return ["Error"], "", f"LLM Auth/Permission Error: {e}"
            if attempt < max_retries - 1: time.sleep(retry_delay); continue
            else: return ["Error"], "", f"LLM Error after {max_retries} attempts: {e}"
    return ["Error"], llm_full_output, "Max retries exceeded for LLM API request."


# --- Policy Violation Check ---
def check_policy_violation(list_of_llm_categories, felony_dates_str_list=None, release_date_str=None):
    if not list_of_llm_categories or "Error" in list_of_llm_categories:
        return False, "Denied due to error in LLM processing or categorization."
    denied_categories_found = []
    for category in list_of_llm_categories:
        if category in NO_GO_CATEGORIES: denied_categories_found.append(category)
    if denied_categories_found: return False, f"Denied based on LLM category: '{denied_categories_found[0]}'" 
    valid_categories_identified_by_llm = [cat for cat in list_of_llm_categories if cat != "Error"]
    if len(valid_categories_identified_by_llm) >= 2:
        return False, "Denied: Description details two or more distinct felony types."
    error_message = "Invalid date format. Please use foundry-MM-DD, foundry-MM, foundry, MM/DD/YYYY, or MM/YYYY (and - variations)."
    felony_datetimes = []
    if felony_dates_str_list:
        for date_str in felony_dates_str_list:
            parsed_date = parse_flexible_date(date_str)
            if parsed_date: felony_datetimes.append(parsed_date)
            elif date_str: return False, f"{error_message} (Offending value: '{date_str}')"
    today = datetime.today()
    if felony_datetimes:
        for date_obj in felony_datetimes:
            if date_obj > (today - timedelta(days=365)): return False, "Denied: Felony conviction within the last year."
    release_date = None
    if release_date_str and release_date_str.strip():
        release_date = parse_flexible_date(release_date_str)
        if not release_date: return False, f"{error_message} (Offending value: '{release_date_str}')"
        if release_date > (today - timedelta(days=365)): return False, "Denied: Released from jail within the last year."
    if felony_datetimes and len(felony_datetimes) >= 2:
        most_recent_felony_date = max(felony_datetimes)
        if most_recent_felony_date > (today - timedelta(days=10 * 365.25)):
            return False, "Denied: Two or more felony convictions (based on dates provided), and the most recent is within the last 10 years."
    return True, "Approved: No policy violations based on LLM category(s) and date checks."

# --- Helper for Critical Keyword Check ---
def check_input_for_critical_keywords(felony_input):
    if not felony_input: return None
    input_lower = felony_input.lower()
    found_critical_categories = set()
    for category, keywords in CRITICAL_INPUT_KEYWORDS.items():
        for keyword in keywords:
            # Apply word boundaries to all critical keywords for more precision
            if re.search(r'\b' + re.escape(keyword) + r'\b', input_lower):
                found_critical_categories.add(category)
                break # Move to next category once one keyword matches for this category
    return list(found_critical_categories) if found_critical_categories else []

# --- Function to Add to Log ---
def add_to_log(log_entry):
    global application_log 
    if len(application_log) >= MAX_LOG_ENTRIES: 
        application_log.pop(0) 
    application_log.append(log_entry)

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index_route(): 
    result_status, reason_message, llm_full_text = None, "", ""
    llm_assigned_categories_display = [] 
    weapon_involved_checked_on_form, weapon_detected_in_text_on_form = False, False
    final_decision_cats_display, dates_extracted_from_text_display = [], []
    log_entry_details = {} 

    if request.method == 'POST':
        felony_input = request.form.get('felony', '') 
        felony_dates_from_box_str = request.form.get('felony_dates', '')
        release_date_input = request.form.get('release_date', '')
        weapon_checkbox_val = request.form.get('weapon_involved')
        weapon_checked_by_user = True if weapon_checkbox_val == 'on' else False

        log_entry_details['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry_details['felony_input'] = felony_input
        log_entry_details['weapon_checkbox'] = weapon_checked_by_user
        log_entry_details['felony_dates_box'] = felony_dates_from_box_str
        log_entry_details['release_date_box'] = release_date_input

        weapon_involved_checked_on_form = weapon_checked_by_user 
        weapon_found_in_text = text_mentions_weapon(felony_input)
        weapon_detected_in_text_on_form = weapon_found_in_text 
        log_entry_details['weapon_in_text'] = weapon_found_in_text

        dates_extracted_from_text = extract_dates_from_text(felony_input)
        dates_extracted_from_text_display = dates_extracted_from_text 
        log_entry_details['dates_from_text'] = dates_extracted_from_text

        felony_dates_from_box_list = []
        if felony_dates_from_box_str:
            felony_dates_from_box_list = [date.strip() for date in felony_dates_from_box_str.split(',') if date.strip()]

        all_felony_dates_str_list = list(set(felony_dates_from_box_list + dates_extracted_from_text))
        log_entry_details['combined_felony_dates_used'] = all_felony_dates_str_list
        print(f"Dates from box: {felony_dates_from_box_list}, Dates from text: {dates_extracted_from_text}, Combined: {all_felony_dates_str_list}")

        if not felony_input:
            result_status, reason_message = "Error", "Felony description cannot be empty."
        else:
            list_of_llm_categories, full_output, error_msg_from_llm = query_gemini_for_offense_analysis(
                felony_input, 
                weapon_checked_by_user=weapon_checked_by_user,
                weapon_found_in_text=weapon_found_in_text
            )
            llm_full_text = full_output
            llm_assigned_categories_display = list_of_llm_categories 
            log_entry_details['llm_assigned_categories'] = list_of_llm_categories
            log_entry_details['llm_full_output'] = full_output

            final_decision_categories = list(list_of_llm_categories) 

            temp_final_categories = []
            description_mentions_violence = text_mentions_violence(felony_input) 
            description_lower = felony_input.lower() 

            for cat in final_decision_categories: 
                current_cat = cat
                is_just_weapon_possession = "possession" in description_lower and weapon_found_in_text and not description_mentions_violence

                if cat == "Violent Crime Involving a Weapon":
                    if is_just_weapon_possession:
                        print(f"Override: LLM said '{cat}', but description is weapon possession without violence. Changing to 'Other'.")
                        current_cat = "Other"
                    elif not weapon_checked_by_user and not weapon_found_in_text: 
                        print(f"Override: LLM said '{cat}', but no weapon signal from user/text & not clearly just possession. Changing to 'Other'.")
                        current_cat = "Other"

                elif cat == "Other" and (weapon_checked_by_user or weapon_found_in_text) and description_mentions_violence:
                    print(f"Override: LLM said 'Other', but weapon signaled AND violence in text. Changing to 'Violent Crime Involving a Weapon'.")
                    current_cat = "Violent Crime Involving a Weapon"

                elif cat == "Distribution Drug Related Crime":
                    is_just_drug_possession = "possession" in description_lower and \
                                             not any(dist_kw in description_lower for dist_kw in 
                                                     ["sell", "distribute", "manufacture", "transport", 
                                                      "traffic", "deliver", "cultivate", "intent to"])
                    if is_just_drug_possession: 
                        print(f"Override: LLM said '{cat}', but input seems like drug possession only. Changing to 'Other'.")
                        current_cat = "Other"
                temp_final_categories.append(current_cat)
            final_decision_categories = temp_final_categories
            final_decision_cats_display = final_decision_categories
            log_entry_details['final_decision_categories'] = final_decision_categories


            if error_msg_from_llm or "Error" in final_decision_categories:
                result_status, reason_message = "Error", error_msg_from_llm or "LLM failed to categorize or an error occurred."
            else:
                is_approved, policy_reason = check_policy_violation(
                    final_decision_categories, 
                    all_felony_dates_str_list, 
                    release_date_input
                )

                if is_approved: 
                    critical_keyword_categories_found = check_input_for_critical_keywords(felony_input)
                    log_entry_details['critical_keywords_in_input'] = critical_keyword_categories_found
                    if critical_keyword_categories_found:
                        for critical_cat in critical_keyword_categories_found:
                            is_already_denied_for_similar = any(fc_cat == critical_cat for fc_cat in final_decision_categories if fc_cat in NO_GO_CATEGORIES)
                            if not is_already_denied_for_similar or "Other" in final_decision_categories :
                                print(f"Keyword Safety Net Triggered: Found '{critical_cat}' related keyword. Overriding Approval.")
                                is_approved = False
                                policy_reason = f"Denied based on keyword check of input (found term related to '{critical_cat}')."
                                log_entry_details['keyword_override_reason'] = policy_reason
                                break 

                result_status, reason_message = ("Approved" if is_approved else "Denied"), policy_reason

        log_entry_details['final_result'] = result_status
        log_entry_details['final_reason'] = reason_message
        add_to_log(log_entry_details) 

    felony_description_val = request.form.get('felony', '') if request.method == 'POST' else ''
    felony_dates_val = request.form.get('felony_dates', '') if request.method == 'POST' else ''
    release_date_val = request.form.get('release_date', '') if request.method == 'POST' else ''

    return render_template('index.html', 
                           result=result_status, reason=reason_message, 
                           llm_output=llm_full_text, 
                           llm_categories_display=llm_assigned_categories_display, 
                           final_categories_display=final_decision_cats_display, 
                           felony_description_val=felony_description_val,
                           felony_dates_val=felony_dates_val, release_date_val=release_date_val,
                           weapon_involved_val=weapon_involved_checked_on_form,
                           weapon_in_text_val=weapon_detected_in_text_on_form,
                           dates_from_text_val=dates_extracted_from_text_display)


@app.route('/check_felony', methods=['POST'])
def check_felony_api():
    felony_input = request.form.get('felony','')
    felony_dates_from_box_str = request.form.get('felony_dates','')
    release_date_input = request.form.get('release_date','')
    weapon_checkbox_val = request.form.get('weapon_involved') 
    weapon_checked_by_user = True if weapon_checkbox_val == 'on' else False

    log_entry_details = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source': 'API',
        'felony_input': felony_input,
        'weapon_checkbox': weapon_checked_by_user,
        'felony_dates_box': felony_dates_from_box_str,
        'release_date_box': release_date_input
    }

    weapon_found_in_text = text_mentions_weapon(felony_input)
    description_mentions_violence_api = text_mentions_violence(felony_input)
    description_lower_api = felony_input.lower()
    log_entry_details['weapon_in_text'] = weapon_found_in_text
    log_entry_details['description_mentions_violence'] = description_mentions_violence_api


    dates_extracted_from_text = extract_dates_from_text(felony_input)
    log_entry_details['dates_from_text'] = dates_extracted_from_text
    felony_dates_from_box_list = []
    if felony_dates_from_box_str:
        felony_dates_from_box_list = [date.strip() for date in felony_dates_from_box_str.split(',') if date.strip()]
    all_felony_dates_str_list = list(set(felony_dates_from_box_list + dates_extracted_from_text))
    log_entry_details['combined_felony_dates_used'] = all_felony_dates_str_list


    if not felony_input: 
        log_entry_details['final_result'] = "Error"
        log_entry_details['final_reason'] = "No felony description provided."
        add_to_log(log_entry_details)
        return jsonify({"status": "Error", "reason": "No felony description provided."})

    list_of_llm_categories, full_output, error_msg_from_llm = query_gemini_for_offense_analysis(
        felony_input, weapon_checked_by_user=weapon_checked_by_user, weapon_found_in_text=weapon_found_in_text
    )
    llm_assigned_categories_for_api = list(list_of_llm_categories)
    final_decision_categories = list(list_of_llm_categories)
    log_entry_details['llm_assigned_categories'] = llm_assigned_categories_for_api
    log_entry_details['llm_full_output'] = full_output


    temp_final_categories = []
    for cat in final_decision_categories:
        current_cat = cat
        is_just_weapon_possession_api = "possession" in description_lower_api and weapon_found_in_text and not description_mentions_violence_api
        if cat == "Violent Crime Involving a Weapon":
            if is_just_weapon_possession_api: current_cat = "Other"
            elif not weapon_checked_by_user and not weapon_found_in_text: current_cat = "Other"
        elif cat == "Other" and (weapon_checked_by_user or weapon_found_in_text) and description_mentions_violence_api: 
            current_cat = "Violent Crime Involving a Weapon"
        elif cat == "Distribution Drug Related Crime":
            is_just_drug_possession = "possession" in description_lower_api and not any(dist_kw in description_lower_api for dist_kw in ["sell", "distribute", "manufacture", "transport", "traffic", "deliver", "cultivate", "intent to"])
            if is_just_drug_possession: current_cat = "Other"
        temp_final_categories.append(current_cat)
    final_decision_categories = temp_final_categories
    log_entry_details['final_decision_categories'] = final_decision_categories


    if error_msg_from_llm or "Error" in final_decision_categories:
        log_entry_details['final_result'] = "Error"
        log_entry_details['final_reason'] = error_msg_from_llm or "LLM failed to categorize."
        add_to_log(log_entry_details)
        return jsonify({"status": "Error", "reason": error_msg_from_llm or "LLM failed to categorize.", 
                        "llm_categories_by_llm": llm_assigned_categories_for_api, 
                        "final_categories_used_for_decision": final_decision_categories,
                        "dates_from_text": dates_extracted_from_text, "llm_full_output": full_output})
    else:
        is_approved, policy_reason = check_policy_violation(final_decision_categories, all_felony_dates_str_list, release_date_input)
        final_status, final_reason, keyword_override_reason = ("Approved" if is_approved else "Denied"), policy_reason, None

        if is_approved:
            critical_keyword_categories_found = check_input_for_critical_keywords(felony_input)
            log_entry_details['critical_keywords_in_input'] = critical_keyword_categories_found
            if critical_keyword_categories_found:
                for critical_cat in critical_keyword_categories_found:
                    is_already_denied_for_similar = any(fc_cat == critical_cat for fc_cat in final_decision_categories if fc_cat in NO_GO_CATEGORIES)
                    if not is_already_denied_for_similar or "Other" in final_decision_categories:
                        final_status, is_approved = "Denied", False
                        keyword_override_reason = f"Denied by input keyword check (found '{critical_cat}')."
                        final_reason = keyword_override_reason
                        log_entry_details['keyword_override_reason'] = keyword_override_reason
                        break

        log_entry_details['final_result'] = final_status
        log_entry_details['final_reason'] = final_reason
        add_to_log(log_entry_details)

        return jsonify({
            "status": final_status, "is_approved": is_approved, "reason": final_reason,
            "llm_categories_by_llm": llm_assigned_categories_for_api, 
            "final_categories_used_for_decision": final_decision_categories,
            "keyword_override_reason": keyword_override_reason,
            "weapon_checkbox_status": weapon_checked_by_user,
            "weapon_detected_in_text": weapon_found_in_text,
            "description_mentions_violence": description_mentions_violence_api, 
            "dates_from_text": dates_extracted_from_text,
            "dates_from_box": felony_dates_from_box_list,
            "combined_dates_used": all_felony_dates_str_list,
            "llm_full_output": full_output
        })

# --- New Log Route ---
@app.route('/log')
def view_log():
    # Ensure application_log is accessible here as well
    global application_log
    return render_template('log.html', logs=list(reversed(application_log)))


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
