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
# Use a global declaration within functions that modify it
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
        # Switched to 'gemini-2.5-flash' as 'gemini-1.5-flash' was also not found.
        GEMINI_MODEL_NAME = 'gemini-2.5-flash' 
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
    "Human Trafficking",
    "Domestic Violence Related Crime" # New category added
]

# --- Define Critical Keywords for Input Scan (Safety Net) ---
CRITICAL_INPUT_KEYWORDS = {
    "Violent Crime Involving a Weapon": [
        # Keywords implying violence + weapon
        "armed robbery", "assault with a deadly weapon", 
        "stabbing with", "shooting at", "shot someone", "fired a gun at", 
        "brandished a weapon during", 
        "used a gun in a robbery", "used a knife to threaten",
        "explosive device", "bombing", "aggravated assault with",
        "aggravated battery with"
    ],
    "Sexual Related Crime": ["sexual assault", "rape", "molestation", "child pornography", "non-consensual sex", "involuntary sex", "sexual battery", "lewd", "lascivious"],
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
        "abduction of a child", "child abduction", "kidnapping a child", "kidnapping of a minor", "abduction"
    ],
    "Domestic Violence Related Crime": [ # New category keywords
        "domestic violence", "domestic abuse", "spousal abuse", 
        "family violence", "domestic battery", "domestic assault",
        "violation of a protective order", "violating restraining order"
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
    "killed", "wounded", "victimized", "abduction", "kidnap", "kidnapping",
    "domestic violence", "domestic abuse", "spousal abuse", "domestic battery", "domestic assault" # New additions
]

# --- Dismissal Keywords ---
DISMISSAL_KEYWORDS = [
    "dropped", "dismissed", "expunged", "sealed", "vacated", 
    "set aside", "adjudication withheld", "nolle prosequi", "nollied",
    "pending dismissal", "being thrown out", "thrown out", 
    "charges dropped", "case dropped", "acquitted", "not guilty"
]

# --- Vague Keywords (for "Needs more info" check) ---
VAGUE_KEYWORDS = [
    "aiding", "abetting", "conspiracy", "accessory", "solicitation",
    "attempt to commit", # Vague without the underlying crime
    "statute", "code", "section", "law", # References to legal text without context
    r'felony\s+[0-9]', r'class\s+[a-z0-9]' # e.g., "felony 3", "class a", "class 3"
]

def text_mentions_weapon(description):
    """
    Checks if the description contains weapon keywords, but ONLY if explicit negations are absent.
    """
    if not description: return False
    description_lower = description.lower()

    # Check for explicit negations of weapon first
    negation_patterns = [
        r'\bno\s+weapon\b',
        r'\bwithout\s+a\s+weapon\b', 
        r'\bwithout\s+weapon\b', 
        r'\bnot\s+armed\b',
        r'\bunarmed\b'
    ]
    if any(re.search(neg_pattern, description_lower) for neg_pattern in negation_patterns):
        return False # Explicitly states no weapon

    # If no negations, check for weapon keywords
    for keyword in WEAPON_KEYWORDS:
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

def text_mentions_dismissal(description):
    """Checks if the description contains keywords indicative of a dismissal or dropped charge."""
    if not description: return False
    description_lower = description.lower()
    for keyword in DISMISSAL_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', description_lower):
            return True
    return False

def text_is_inherently_vague(description):
    """Checks if the description contains keywords that are inherently vague."""
    if not description: return False
    description_lower = description.lower()
    # Check for vague keywords
    for keyword_pattern in VAGUE_KEYWORDS:
        if re.search(r'\b' + keyword_pattern + r'\b', description_lower, re.IGNORECASE):
            # Check if it's ONLY vague keywords and not specific crimes
            # This avoids flagging "conspiracy to commit murder" as vague
            has_critical_keyword = any(
                re.search(r'\b' + re.escape(kw) + r'\b', description_lower) 
                for category_keywords in CRITICAL_INPUT_KEYWORDS.values() 
                for kw in category_keywords
            )
            if not has_critical_keyword:
                return True # It's vague and doesn't contain a specific critical crime keyword
    return False


# --- Super Flexible Date Parsing Helper ---
def parse_flexible_date(date_str):
    """
    Parses a date string in various formats (YYYY-MM-DD, MM/DD/YYYY, YYYY-MM, MM/YYYY, YYYY, Month YYYY).
    Defaults partial dates to the end of that period (month or year).
    """
    date_str = date_str.strip()
    print(f"DEBUG: Parsing date string: '{date_str}'") # Debug print

    # --- FIX: Explicit 4-digit year check ---
    # Try to match YYYY format first if it's just 4 digits
    if re.fullmatch(r'\b([12]\d{3})\b', date_str):
        print("DEBUG: Matched YYYY only format.")
        try:
            dt = datetime.strptime(date_str, '%Y')
            return dt.replace(month=12, day=31)
        except ValueError:
            print(f"DEBUG: Failed to parse {date_str} as YYYY even after fullmatch.")
            pass # Should not happen, but good practice

    # Normalize month names
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

    # Define formats: (format_string, is_partial)
    formats_to_try = [
        ('%Y-%m-%d', False), ('%m/%d/%Y', False), ('%m-%d-%Y', False), # Full dates
        ('%b %d, %Y', False), ('%d %b %Y', False), # Dates with month names
        ('%b %Y', True), ('%B %Y', True), ('%b, %Y', True), ('%B, %Y', True), # Month-Year
        ('%Y-%m', True), ('%m/%Y', True), ('%m-%Y', True), # Month-Year numeric
        ('%Y', True) # Year only
    ]

    for fmt, is_partial in formats_to_try:
        try:
            # Use the normalized string if format expects a month name
            string_to_parse = temp_date_str if '%b' in fmt.lower() or '%B' in fmt.lower() else date_str
            dt = datetime.strptime(string_to_parse, fmt)

            if is_partial:
                # If only month and year given, set to last day of that month
                if fmt in ['%Y-%m', '%m/%Y', '%m-%Y', '%b %Y', '%B %Y', '%b, %Y', '%B, %Y']: 
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    return dt.replace(day=last_day)
                # If only year given, set to last day of that year
                elif fmt == '%Y': 
                    print("DEBUG: Matched %Y format in loop.") # Debug print
                    return dt.replace(month=12, day=31)
            return dt # Return full date
        except ValueError:
            continue # Try next format

    print(f"DEBUG: All formats failed for: '{date_str}'") # Debug print
    return None # Return None if no format matches

# --- Date Extraction from Text ---
def extract_dates_from_text(text_content):
    """Extracts date strings from free text."""
    if not text_content: return []

    found_date_strings = []

    # Patterns for YYYY-MM-DD, MM/DD/YYYY, YYYY-MM, MM/YYYY, Month YYYY, YYYY
    date_patterns = [
        r'\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b', # YYYY-MM-DD
        r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b', # MM/DD/YYYY
        # Month Name Day, Year (e.g., Jan 15, 2023 or January 15 2023)
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4})\b',
        # Day Month Name Year (e.g., 15 Jan 2023)
        r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
        r'\b(\d{4}[-/]\d{1,2})\b',           # YYYY-MM
        r'\b(\d{1,2}[/-]\d{4})\b',           # MM/YYYY
        # Month Name YYYY (e.g., Jan 2023 or January, 2023)
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s+\d{4})\b',
        r'\b([12]\d{3})\b'                   # YYYY (as a standalone 4-digit number)
    ]

    temp_text = text_content

    # Helper to collect match and replace to avoid re-matching
    def replace_and_collect(match_obj, collection):
        collection.append(match_obj.group(0)); 
        return " " * len(match_obj.group(0)) # Replace with spaces

    for pattern in date_patterns:
        # Use re.IGNORECASE for patterns with month names
        flags = re.IGNORECASE if "jan" in pattern.lower() else 0
        temp_text = re.sub(pattern, lambda m: replace_and_collect(m, found_date_strings), temp_text, flags=flags)

    return list(set(found_date_strings)) # Return unique date strings found

# --- LLM Interaction with Multi-Offense Categorization ---
def query_gemini_for_offense_analysis(felony_description, weapon_checked_by_user=None, weapon_found_in_text=False, max_retries=3, retry_delay=5):
    """
    Queries Google Gemini to identify and categorize multiple offenses in a description.
    Returns:
        tuple: (list_of_categories, str_llm_full_output, str_error_message)
               - list_of_categories: A list of categories assigned by the LLM (or ["Error"] / ["Other"]).
               - llm_full_output: The full text response from the LLM.
               - error_message: None if successful, or an error string.
    """
    if not GOOGLE_API_KEY: 
        return ["Error"], "", "Google API key not configured. Please set GOOGLE_API_KEY in environment variables."
    if not gemini_model: 
        return ["Error"], "", "Gemini model not initialized. Check API key, model name, and initial configuration."

    # Create the list of categories for the prompt
    category_list_for_prompt = "\n".join([f"- {cat}" for cat in NO_GO_CATEGORIES])

    # Build the weapon guidance string
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

    weapon_guidance_parts.append('Note: A "weapon" can include traditional arms as well as any object used to inflict or threaten serious harm (e.g., a stick, rock, heavy object, bottle, vehicle used intentionally to harm, shovel) if the context implies its use as such in a violent crime.')
    combined_weapon_guidance = " ".join(weapon_guidance_parts)

    # Construct the full prompt
    prompt = f"""
You are a legal assistant. Analyze the following felony description, which may describe one or more distinct offenses.
Your primary task is to identify each distinct potential felony mentioned and categorize each one based *only* on the information explicitly provided in the description. Do not infer additional charges or circumstances not stated.

Guidance on Weapon Involvement (applies to each offense if relevant):
{combined_weapon_guidance}

Instructions:
1.  Read the entire "Felony Description to analyze".
2.  Identify each distinct potential felony described.
3.  For EACH distinct offense identified, assign it the most fitting category from the "Categories to choose from" list below.
    - For "Violent Crime Involving a Weapon": The description must indicate BOTH a violent act AND the involvement of a weapon in that specific violent act. Mere possession of a weapon, even if a felony, without a described violent act *using* that weapon, should be categorized as "Other".
    - For "Distribution Drug Related Crime": If it's only possession, categorize as "Other". If it's selling, manufacturing, trafficking, etc., use "Distribution Drug Related Crime".
    - For "Human Trafficking": Consider descriptions involving exploitation, smuggling, or abduction of a person (e.g., "child abduction", "kidnapping").
    - For "Domestic Violence Related Crime": Categorize offenses described as domestic violence, domestic abuse, spousal abuse, or felony assault/battery against a family or household member here.
    - If an offense doesn't fit a specific prohibited category, assign "Other".
4.  Format your output for the categories by listing each identified offense and its category. Start each category on a new line, prefixed with "Identified Category: ".
    Example of format for multiple offenses:
    Identified Category: [Category for first offense or Other]
    Identified Category: [Category for second offense or Other]
    (If only one offense, list one "Identified Category:" line. If no specific offenses classifiable into the list are found, output one line: "Identified Category: Other")

5.  After listing all "Identified Category:" lines, if the original description was too vague to determine the specific nature of the underlying felony (e.g., 'aiding and abetting a felony' without specifying the felony, or just a statute number like 'Class 3 Felony' with no details), add a final line formatted EXACTLY as:
    Clarity Check: Vague - More information needed.

6.  Finally, provide a single, concise overall legal explanation of the alleged crimes under U.S. law.

Categories to choose from:
{category_list_for_prompt}
- Other

Felony Description to analyze: "{felony_description}"
    """

    generation_config = genai.types.GenerationConfig(
        temperature=0.15, # Low temperature for more deterministic categorization
        max_output_tokens=1000
    )

    llm_full_output = ""
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check for valid response content
            if response.parts:
                llm_full_output = response.text.strip()

                # --- Parse LLM Output ---
                # Find all lines starting with "Identified Category: "
                identified_categories = re.findall(r"Identified Category:\s*(.+)", llm_full_output, re.IGNORECASE)

                if not identified_categories:
                    # Fallback for "Other" if no specific categories are listed
                    if "Identified Category: Other" in llm_full_output or "Primary Category: Other" in llm_full_output:
                         return ["Other"], llm_full_output, None # Valid case, single "Other"

                    # Fallback for old prompt format (less likely)
                    primary_category_match = re.search(r"Primary Category:\s*(.+)", llm_full_output, re.IGNORECASE) 
                    if primary_category_match:
                        parsed_category = primary_category_match.group(1).strip()
                        for known_cat in (NO_GO_CATEGORIES + ["Other"]):
                            if known_cat.lower() == parsed_category.lower():
                                return [known_cat], llm_full_output, None
                        print(f"Warning: LLM used 'Primary Category' format. Category '{parsed_category}' not known. Treating as ['Other'].")
                        return ["Other"], llm_full_output, None

                    error_msg = "Error: LLM did not provide categories in the expected 'Identified Category:' format."
                    print(f"Gemini API attempt {attempt + 1}: {error_msg} Full output: {llm_full_output[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return ["Error"], llm_full_output, error_msg # Failed after retries

                # Normalize found categories
                normalized_categories = []
                for cat_str in identified_categories:
                    cat_str_clean = cat_str.strip()
                    found_known = False
                    for known_cat in (NO_GO_CATEGORIES + ["Other"]): # Check against "Other" as well
                        if known_cat.lower() == cat_str_clean.lower():
                            normalized_categories.append(known_cat)
                            found_known = True
                            break
                    if not found_known:
                        # If LLM returns a category not in our list (e.g., "Drug Possession"), treat it as "Other"
                        print(f"Warning: LLM returned category '{cat_str_clean}' not in predefined list. Treating as 'Other'.")
                        normalized_categories.append("Other")

                return normalized_categories if normalized_categories else ["Other"], llm_full_output, None # Return list of categories
                # --- End Parse LLM Output ---

            else: 
                # Handle cases where response.parts is empty (e.g., safety blocks)
                error_details = []
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_details.append(f"Prompt blocked due to: {response.prompt_feedback.block_reason.name}")
                if response.candidates:
                    for cand in response.candidates:
                        if cand.finish_reason and cand.finish_reason.name != 'STOP':
                            error_details.append(f"Generation finished due to: {cand.finish_reason.name}")
                        if cand.safety_ratings:
                            problematic_ratings = [(r.category.name, r.probability.name) for r in cand.safety_ratings if r.probability.name not in ['NEGLIGIBLE', 'LOW']]
                            if problematic_ratings:
                                error_details.append(f"Safety Ratings Issues: {problematic_ratings}")

                error_msg = "Error: No content generated by LLM."
                if error_details:
                    error_msg += " Details: " + "; ".join(error_details)
                print(f"Gemini API attempt {attempt + 1} returned no valid content: {error_msg}")

                # Don't retry safety blocks
                if any(block_reason in error_msg.upper() for block_reason in ["SAFETY", "BLOCK_REASON_OTHER"]):
                    return ["Error"], "", error_msg 

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return ["Error"], "", error_msg # Failed after retries

        except Exception as e:
            # Handle API errors like Quota
            if "429" in str(e) and "quota" in str(e).lower():
                print(f"Quota exceeded on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    match_retry = re.search(r"retry_delay\s*\{\s*seconds\s*:\s*(\d+)\s*\}", str(e))
                    dynamic_retry_delay = int(match_retry.group(1)) if match_retry else retry_delay + (attempt * 5)
                    print(f"Retrying in {dynamic_retry_delay} seconds...")
                    time.sleep(dynamic_retry_delay)
                    continue
                else:
                    return ["Error"], "", f"LLM Quota Error after {max_retries} attempts: {e}"

            # Handle other general exceptions
            print(f"General error querying Gemini (attempt {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
            if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e):
                return ["Error"], "", f"LLM Auth/Permission Error: {e}" # Don't retry auth errors

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                return ["Error"], "", f"LLM Error after {max_retries} attempts: {e}"

    return ["Error"], llm_full_output, "Max retries exceeded for LLM API request."


# --- Policy Violation Check ---
def check_policy_violation(list_of_llm_categories, felony_dates_str_list=None, release_date_str=None):
    """
    Checks LLM categories and dates against policy rules.
    Returns: (is_approved, reason_message)
    """

    # Check 1: LLM Categories - any single "No Go" category results in denial
    if not list_of_llm_categories or "Error" in list_of_llm_categories:
        return False, "Denied due to error in LLM processing or categorization."

    denied_categories_found = []
    for category in list_of_llm_categories:
        if category in NO_GO_CATEGORIES:
            denied_categories_found.append(category)

    if denied_categories_found:
        # Return the first "No Go" category found
        return False, f"Denied based on LLM category: '{denied_categories_found[0]}'" 

    # --- CORRECTED: Check if the LLM identified 2 or more distinct felony acts in the text ---
    # Count valid, non-Error categories identified by the LLM.
    # If the list contains 2 or more entries (e.g., ["Other", "Other"]), it means the LLM
    # parsed multiple distinct offenses, even if they were all categorized as "Other".
    valid_categories_identified_by_llm = [cat for cat in list_of_llm_categories if cat != "Error"]

    if len(valid_categories_identified_by_llm) >= 2:
        return False, "Denied: Description details two or more distinct felony types."

    # Check 3: Date-related checks (applied globally to all provided dates)
    # FIX: Corrected typo in error message
    error_message = "Invalid date format. Please use YYYY-MM-DD, YYYY-MM, YYYY, MM/DD/YYYY, or MM/YYYY (and - variations)."

    felony_datetimes = []
    if felony_dates_str_list:
        for date_str in felony_dates_str_list:
            parsed_date = parse_flexible_date(date_str)
            if parsed_date:
                felony_datetimes.append(parsed_date)
            elif date_str.strip(): # If it's not an empty string but failed to parse
                return False, f"{error_message} (Offending value: '{date_str}')"

    today = datetime.today()

    # Rule: Felony within a year
    if felony_datetimes:
        for date_obj in felony_datetimes:
            # Check if the date is in the future (invalid) or within the last 365 days
            if date_obj > today:
                 return False, f"Denied: Felony conviction date '{date_obj.strftime('%Y-%m-%d')}' is in the future."
            if date_obj > (today - timedelta(days=365)):
                return False, "Denied: Felony conviction within the last year."

    # Rule: Released from jail within a year
    release_date = None
    if release_date_str and release_date_str.strip():
        release_date = parse_flexible_date(release_date_str)
        if not release_date:
            return False, f"{error_message} (Offending value: '{release_date_str}')"
        if release_date > today:
             return False, f"Denied: Release date '{release_date.strftime('%Y-%m-%d')}' is in the future."
        if release_date > (today - timedelta(days=365)):
            return False, "Denied: Released from jail within the last year."

    # Rule: Two or more felonies (based on dates) AND most recent is within 10 years
    if felony_datetimes and len(felony_datetimes) >= 2:
        most_recent_felony_date = max(felony_datetimes)
        # Check if the most recent of these 2+ felonies is within the last 10 years
        if most_recent_felony_date > (today - timedelta(days=10 * 365.25)):
            return False, "Denied: Two or more felony convictions (based on dates provided), and the most recent is within the last 10 years."

    # If all checks pass
    return True, "Approved: No policy violations based on LLM category(s) and date checks."

# --- Helper for Critical Keyword Check ---
def check_input_for_critical_keywords(felony_input):
    """
    Scans input text for critical keywords as a safety net.
    Uses word boundaries (\b) for all keywords.
    """
    if not felony_input: return []
    input_lower = felony_input.lower()

    # Check for negations of violence/weapons *before* checking for weapon keywords
    negation_patterns = [r'\bno\s+weapon\b', r'\bwithout\s+a\s+weapon\b', r'\bwithout\s+weapon\b', r'\bnot\s+armed\b', r'\bunarmed\b']
    if any(re.search(neg_pattern, input_lower) for neg_pattern in negation_patterns):
        categories_to_skip_for_keywords = ["Violent Crime Involving a Weapon"]
    else:
        categories_to_skip_for_keywords = []

    found_critical_categories = set()
    for category, keywords in CRITICAL_INPUT_KEYWORDS.items():
        if category in categories_to_skip_for_keywords:
            continue # Skip weapon keywords if a negation was found

        for keyword in keywords:
            # Use word boundary check for all critical keywords
            if re.search(r'\b' + re.escape(keyword) + r'\b', input_lower):
                found_critical_categories.add(category)
                break # Move to the next category once one keyword is found

    return list(found_critical_categories) if found_critical_categories else []

# --- Function to Add to Log ---
def add_to_log(log_entry):
    """Adds a new log entry to the global log, ensuring thread safety and size limit."""
    global application_log # Explicitly declare intent to modify the global variable
    try:
        # In a multi-threaded server, you might add a lock here if needed,
        # but for simple list append/pop(0), Python's GIL often provides enough safety.
        if len(application_log) >= MAX_LOG_ENTRIES: 
            application_log.pop(0) # Remove the oldest entry
        application_log.append(log_entry)
    except Exception as e:
        print(f"Error adding to log: {e}")

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index_route(): 
    """Handles the main web page logic."""
    result_status, reason_message, llm_full_text = None, "", ""
    llm_assigned_categories_display = [] 
    weapon_involved_checked_on_form, weapon_detected_in_text_on_form = False, False
    final_decision_cats_display, dates_extracted_from_text_display = [], []
    log_entry_details = {} # Prepare log entry

    if request.method == 'POST':
        # --- 1. Collect All Inputs ---
        felony_input = request.form.get('felony', '') 
        felony_dates_from_box_str = request.form.get('felony_dates', '')
        release_date_input = request.form.get('release_date', '')
        weapon_checkbox_val = request.form.get('weapon_involved')
        weapon_checked_by_user = True if weapon_checkbox_val == 'on' else False

        # Log initial inputs
        log_entry_details = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Web UI',
            'felony_input': felony_input,
            'weapon_checkbox': weapon_checked_by_user,
            'felony_dates_box': felony_dates_from_box_str,
            'release_date_box': release_date_input
        }

        # --- 2. Process Inputs ---
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

        description_is_vague = text_is_inherently_vague(felony_input)
        log_entry_details['input_is_vague'] = description_is_vague

        # --- 3. Initial Policy Checks (Before LLM) ---
        if not felony_input:
            result_status, reason_message = "Error", "Felony description cannot be empty."
        elif description_is_vague:
             result_status = "Review Required"
             reason_message = "Need more information: The description is too vague (e.g., 'Class 3 felony' or 'aiding and abetting' without specifying the crime). Please provide the specific felony charge."
             llm_full_text = "LLM not called due to inherently vague input." 
        else:
            # --- 4. Call LLM ---
            list_of_llm_categories, full_output, error_msg_from_llm = query_gemini_for_offense_analysis(
                felony_input, 
                weapon_checked_by_user=weapon_checked_by_user,
                weapon_found_in_text=weapon_found_in_text
            )
            llm_full_text = full_output
            llm_assigned_categories_display = list(list_of_llm_categories) 
            log_entry_details['llm_assigned_categories'] = list(list_of_llm_categories)
            log_entry_details['llm_full_output'] = full_output

            # --- 5. Apply Overrides & Business Logic ---
            final_decision_categories = list(list_of_llm_categories) # Start with LLM's categories

            # Apply override logic
            temp_final_categories = []
            description_mentions_violence = text_mentions_violence(felony_input) 
            description_lower = felony_input.lower() 

            for cat in final_decision_categories: 
                current_cat = cat

                # Check for "Possession of weapon" vs "Violent crime"
                is_just_weapon_possession = ("possession" in description_lower and weapon_found_in_text and not description_mentions_violence)

                if cat == "Violent Crime Involving a Weapon":
                    # Override: Downgrade to "Other" if it's just possession
                    if is_just_weapon_possession:
                        print("Override: LLM said 'Violent Crime w/ Weapon', but input looks like possession only. Downgrading to 'Other'.")
                        current_cat = "Other"
                    # Override: Downgrade to "Other" if LLM said weapon, but user/text scan finds no signal (and no explicit negation)
                    elif not weapon_checked_by_user and not weapon_found_in_text: 
                        print(f"Override: LLM said 'Violent Crime w/ Weapon', but checkbox/text scan found no weapon. Downgrading to 'Other'.")
                        current_cat = "Other"

                elif cat == "Other" and (weapon_checked_by_user or weapon_found_in_text) and description_mentions_violence:
                    # Override: Upgrade "Other" to "Violent" if weapon is signaled AND violence is present
                    explicit_no_weapon_in_text = any(re.search(neg_pattern, description_lower) for neg_pattern in [r'\bno\s+weapon\b', r'\bwithout\s+a\s+weapon\b', r'\bwithout\s+weapon\b', r'\bunarmed\b'])
                    if not explicit_no_weapon_in_text: # Only upgrade if text doesn't explicitly say "no weapon"
                        print(f"Override: LLM said 'Other', but weapon signaled AND violence in text. Upgrading to 'Violent Crime Involving a Weapon'.")
                        current_cat = "Violent Crime Involving a Weapon"
                    else:
                        print(f"Notice: LLM said 'Other', weapon keyword found but also negated in text. Violence present. Keeping 'Other'.")
                        # current_cat remains "Other"

                elif cat == "Distribution Drug Related Crime":
                    # Override: Downgrade to "Other" if it's just possession
                    is_just_drug_possession = "possession" in description_lower and \
                                             not any(re.search(r'\b' + re.escape(dist_kw) + r'\b', description_lower) for dist_kw in 
                                                     ["sell", "distribute", "manufacture", "transport", 
                                                      "traffic", "deliver", "cultivate", "intent to"])
                    if is_just_drug_possession: 
                        print("Override: LLM said 'Distribution', but input looks like possession only. Downgrading to 'Other'.")
                        current_cat = "Other"

                temp_final_categories.append(current_cat)

            final_decision_categories = temp_final_categories
            final_decision_cats_display = final_decision_categories
            log_entry_details['final_decision_categories'] = final_decision_categories

            # --- 6. Final Decision Logic ---
            description_mentions_dismissal = text_mentions_dismissal(felony_input)
            log_entry_details['mentions_dismissal'] = description_mentions_dismissal

            # Check if any final category is a "No Go"
            is_no_go_offense_present = any(cat in NO_GO_CATEGORIES for cat in final_decision_categories)

            # Check if critical keywords found categories that the LLM/overrides missed
            critical_keyword_categories_found = check_input_for_critical_keywords(felony_input)
            log_entry_details['critical_keywords_found'] = critical_keyword_categories_found

            # A keyword is critical if it maps to a "No Go" category that isn't already in our final list
            is_critical_keyword_present = any(
                cat in NO_GO_CATEGORIES and cat not in final_decision_categories 
                for cat in critical_keyword_categories_found
            )

            # --- RE-ORDERED LOGIC ---

            # Rule 1: Pending Dismissal check (This overrides a denial)
            if (is_no_go_offense_present or is_critical_keyword_present) and description_mentions_dismissal:
                result_status = "Pending Proof"
                reason_message = "Denied until dropped/dismissed: A disqualifying offense was mentioned but may have been dropped or dismissed. Manual verification of court records is required."

            # Rule 2: LLM Error check
            elif error_msg_from_llm or "Error" in final_decision_categories:
                result_status, reason_message = "Error", error_msg_from_llm or "LLM failed to categorize or an error occurred."

            # Rule 3: Check for "No Go" categories (from LLM or overrides)
            elif is_no_go_offense_present:
                 denied_category = next(cat for cat in final_decision_categories if cat in NO_GO_CATEGORIES)
                 result_status, reason_message = "Denied", f"Denied based on LLM category: '{denied_category}'"

            # Rule 4: Critical Keyword Safety Net (if LLM missed it)
            elif is_critical_keyword_present:
                 missed_category = next(cat for cat in critical_keyword_categories_found if cat in NO_GO_CATEGORIES and cat not in final_decision_categories)
                 result_status, reason_message = "Denied", f"Denied based on keyword check of input (found term related to '{missed_category}')."

            # Rule 5: Main Policy Check (Dates, Multi-offense count)
            # This runs if no specific "No Go" category was found
            else:
                is_approved_by_policy, policy_reason = check_policy_violation(
                    final_decision_categories, 
                    all_felony_dates_str_list, 
                    release_date_input
                )

                if not is_approved_by_policy:
                    # This catches date violations or the "2 or more distinct felony types" rule
                    result_status, reason_message = "Denied", policy_reason
                else:
                    # --- This is the ONLY case where it can be Approved OR Vague ---
                    # Rule 6: Vague check (from LLM) - ONLY if it's not denied by anything else
                    if "Clarity Check: Vague" in llm_full_text:
                         result_status = "Review Required"
                         reason_message = "Need more information: The specific nature of the underlying felony could not be determined from the description provided."
                    else:
                        # Rule 7: All checks passed!
                        result_status, reason_message = "Approved", policy_reason # policy_reason will be "Approved: No policy violations..."

        # --- 7. Log Final Decision ---
        log_entry_details['final_result'] = result_status
        log_entry_details['final_reason'] = reason_message
        add_to_log(log_entry_details) 

    # --- 8. Prepare values for re-rendering the form ---
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
    """
    Handles the API endpoint logic.
    This logic MUST be kept in sync with the index_route POST handler.
    """
    start_time = time.time()

    # --- 1. Collect All Inputs ---
    felony_input = request.form.get('felony', '') 
    felony_dates_from_box_str = request.form.get('felony_dates', '')
    release_date_input = request.form.get('release_date', '')
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

    # --- 2. Process Inputs ---
    weapon_found_in_text = text_mentions_weapon(felony_input)
    log_entry_details['weapon_in_text'] = weapon_found_in_text
    dates_extracted_from_text = extract_dates_from_text(felony_input)
    log_entry_details['dates_from_text'] = dates_extracted_from_text
    felony_dates_from_box_list = []
    if felony_dates_from_box_str:
        felony_dates_from_box_list = [date.strip() for date in felony_dates_from_box_str.split(',') if date.strip()]
    all_felony_dates_str_list = list(set(felony_dates_from_box_list + dates_extracted_from_text))
    log_entry_details['combined_felony_dates_used'] = all_felony_dates_str_list
    description_is_vague = text_is_inherently_vague(felony_input)
    log_entry_details['input_is_vague'] = description_is_vague

    result_status, reason_message, llm_full_text = None, "", ""
    list_of_llm_categories = []
    final_decision_categories = []

    # --- 3. Initial Policy Checks (Before LLM) ---
    if not felony_input:
        result_status, reason_message = "Error", "Felony description cannot be empty."
    elif description_is_vague:
         result_status = "Review Required"
         reason_message = "Need more information: The description is too vague (e.g., 'Class 3 felony' or 'aiding and abetting' without specifying the crime). Please provide the specific felony charge."
         llm_full_text = "LLM not called due to inherently vague input." 
    else:
        # --- 4. Call LLM ---
        list_of_llm_categories, full_output, error_msg_from_llm = query_gemini_for_offense_analysis(
            felony_input, 
            weapon_checked_by_user=weapon_checked_by_user,
            weapon_found_in_text=weapon_found_in_text
        )
        llm_full_text = full_output
        log_entry_details['llm_assigned_categories'] = list_of_llm_categories
        log_entry_details['llm_full_output'] = full_output

        # --- 5. Apply Overrides & Business Logic ---
        final_decision_categories = list(list_of_llm_categories)
        temp_final_categories = []
        description_mentions_violence = text_mentions_violence(felony_input) 
        description_lower = felony_input.lower() 
        for cat in final_decision_categories: 
            current_cat = cat
            is_just_weapon_possession = ("possession" in description_lower and weapon_found_in_text and not description_mentions_violence)
            if cat == "Violent Crime Involving a Weapon":
                if is_just_weapon_possession: current_cat = "Other"
                elif not weapon_checked_by_user and not weapon_found_in_text: current_cat = "Other"
            elif cat == "Other" and (weapon_checked_by_user or weapon_found_in_text) and description_mentions_violence:
                explicit_no_weapon_in_text = any(re.search(neg_pattern, description_lower) for neg_pattern in [r'\bno\s+weapon\b', r'\bwithout\s+a\s+weapon\b', r'\bwithout\s+weapon\b', r'\bunarmed\b'])
                if not explicit_no_weapon_in_text: current_cat = "Violent Crime Involving a Weapon"
            elif cat == "Distribution Drug Related Crime":
                is_just_drug_possession = "possession" in description_lower and \
                                         not any(re.search(r'\b' + re.escape(dist_kw) + r'\b', description_lower) for dist_kw in 
                                                 ["sell", "distribute", "manufacture", "transport", 
                                                  "traffic", "deliver", "cultivate", "intent to"])
                if is_just_drug_possession: current_cat = "Other"
            temp_final_categories.append(current_cat)
        final_decision_categories = temp_final_categories
        log_entry_details['final_decision_categories'] = final_decision_categories

        # --- 6. Final Decision Logic (RE-ORDERED) ---
        description_mentions_dismissal = text_mentions_dismissal(felony_input)
        log_entry_details['mentions_dismissal'] = description_mentions_dismissal

        is_no_go_offense_present = any(cat in NO_GO_CATEGORIES for cat in final_decision_categories)

        critical_keyword_categories_found = check_input_for_critical_keywords(felony_input)
        log_entry_details['critical_keywords_found'] = critical_keyword_categories_found

        is_critical_keyword_present = any(
            cat in NO_GO_CATEGORIES and cat not in final_decision_categories 
            for cat in critical_keyword_categories_found
        )

        # Rule 1: Pending Dismissal check (This overrides a denial)
        if (is_no_go_offense_present or is_critical_keyword_present) and description_mentions_dismissal:
            result_status = "Pending Proof"
            reason_message = "Denied until dropped/dismissed: A disqualifying offense was mentioned but may have been dropped or dismissed. Manual verification of court records is required."

        # Rule 2: LLM Error check
        elif error_msg_from_llm or "Error" in final_decision_categories:
            result_status, reason_message = "Error", error_msg_from_llm or "LLM failed to categorize or an error occurred."

        # Rule 3: Check for "No Go" categories (from LLM or overrides)
        elif is_no_go_offense_present:
             denied_category = next(cat for cat in final_decision_categories if cat in NO_GO_CATEGORIES)
             result_status, reason_message = "Denied", f"Denied based on LLM category: '{denied_category}'"

        # Rule 4: Critical Keyword Safety Net (if LLM missed it)
        elif is_critical_keyword_present:
             missed_category = next(cat for cat in critical_keyword_categories_found if cat in NO_GO_CATEGORIES and cat not in final_decision_categories)
             result_status, reason_message = "Denied", f"Denied based on keyword check of input (found term related to '{missed_category}')."

        # Rule 5: Main Policy Check (Dates, Multi-offense count)
        # This runs if no specific "No Go" category was found
        else:
            is_approved_by_policy, policy_reason = check_policy_violation(
                final_decision_categories, 
                all_felony_dates_str_list, 
                release_date_input
            )

            if not is_approved_by_policy:
                # This catches date violations or the "2 or more distinct felony types" rule
                result_status, reason_message = "Denied", policy_reason
            else:
                # --- This is the ONLY case where it can be Approved OR Vague ---
                # Rule 6: Vague check (from LLM) - ONLY if it's not denied by anything else
                if "Clarity Check: Vague" in llm_full_text:
                     result_status = "Review Required"
                     reason_message = "Need more information: The specific nature of the underlying felony could not be determined from the description provided."
                else:
                    # Rule 7: All checks passed!
                    result_status, reason_message = "Approved", policy_reason # policy_reason will be "Approved: No policy violations..."

    # --- 7. Log Final Decision & Respond ---
    log_entry_details['final_result'] = result_status
    log_entry_details['final_reason'] = reason_message
    log_entry_details['processing_time_ms'] = (time.time() - start_time) * 1000
    add_to_log(log_entry_details)

    response_data = {
        "result": result_status,
        "reason": reason_message,
        "llm_assigned_categories": list_of_llm_categories,
        "final_decision_categories": final_decision_categories,
        "llm_full_output": llm_full_text,
        "debug_info": {
            "weapon_checkbox_was": "Checked" if weapon_checked_by_user else "Unchecked",
            "weapon_in_text_was": "Yes" if weapon_found_in_text else "No",
            "dates_extracted_from_text": dates_extracted_from_text,
            "inherently_vague_input_check": "Yes" if description_is_vague else "No",
            "dismissal_keywords_found": "Yes" if description_mentions_dismissal else "No"
        }
    }
    response = jsonify(response_data)
    response.headers['Content-Type'] = 'application/json'
    return response


# --- New Log Route ---
@app.route('/log')
def view_log():
    """Renders the log page."""
    global application_log
    # Pass a reversed copy to the template so newest are first
    return render_template('log.html', logs=list(reversed(application_log)))


# --- Main Execution ---
if __name__ == '__main__':
    # Gunicorn (used by Render) will run the 'app' object directly.
    # This block is for local development (e.g., python main.py)
    port = int(os.environ.get('PORT', 8080))
    # Set debug=False for production, True for local development
    # Render sets its own environment, so debug=True here is fine for testing.
    app.run(debug=True, host='0.0.0.0', port=port)
"

