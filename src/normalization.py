import re
import pandas as pd
from typing import List, Dict

# Stopwords common in business names
BUSINESS_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co', 'company', 
    'and', 'or', 'of', 'the', 'group', 'holdings', 'international', 'services', 'enterprises',
    'pvt', 'private', 'llp', 'gmbh', 'sa', 'spa', 'bv', 'plc', 'lp'
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Replace non-alphanumeric with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Reduce multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_acronym(tokens: List[str]) -> str:
    return "".join([t[0] for t in tokens if len(t) > 0])

def extract_numeric(text: str) -> List[str]:
    return re.findall(r'\d+', str(text))

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Names
    df['name_raw'] = df['business_name'].fillna('')
    df['name_basic'] = df['name_raw'].apply(clean_text)
    
    def process_name_core(basic_name):
        tokens = basic_name.split()
        core_tokens = [t for t in tokens if t not in BUSINESS_STOPWORDS]
        if len(core_tokens) == 0:
            core_tokens = tokens # fallback if name is entirely stopwords
        return " ".join(core_tokens), core_tokens
        
    core_series = df['name_basic'].apply(process_name_core)
    df['name_core'] = core_series.apply(lambda x: x[0])
    df['name_tokens'] = core_series.apply(lambda x: x[1])
    df['name_acronym'] = df['name_tokens'].apply(extract_acronym)
    
    # 2. Addresses
    df['address_raw'] = df['business_address'].fillna('')
    df['address_basic'] = df['address_raw'].apply(clean_text)
    df['address_tokens'] = df['address_basic'].apply(lambda x: x.split())
    
    # 3. Numeric tokens
    df['numeric_tokens'] = (df['name_raw'] + " " + df['address_raw']).apply(extract_numeric)
    
    # 4. Country
    df['country_raw'] = df['country'].fillna('')
    df['country_normalized'] = df['country_raw'].apply(lambda x: str(x).lower().strip())
    
    return df

if __name__ == "__main__":
    # Test normalization
    test_df = pd.DataFrame({
        'business_name': ['Orelee\'s Barbershop Inc.', 'Prime Money LLC'],
        'business_address': ['1795 Westchester Drive, High Point, NC', '17560 Ellis Road, Tahlequah, OK'],
        'country': ['US', 'US']
    })
    norm_df = normalize_dataframe(test_df)
    print(norm_df[['name_raw', 'name_basic', 'name_core', 'name_acronym', 'numeric_tokens']].head())
