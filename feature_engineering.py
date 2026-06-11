import os
import ast
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# My cleaned dataset from the interim phase

src_path = os.path.normpath(os.path.join(
    script_dir, '..', 'Second_Report_Content', 'Cleaned_Data', 'listings_cleaned.csv'
))
out_path = os.path.join(script_dir, 'Cleaned_Data', 'listings_engineered.csv')

# City-centre landmarks (lat, lon). I designated each one to a well-known central square
CITY_CENTRES = {
    'Amsterdam':  (52.3731,  4.8926),   # Dam Square
    'Berlin':     (52.5163, 13.3777),   # Brandenburg Gate
    'Brussels':   (50.8467,  4.3525),   # Grand Place
    'Budapest':   (47.4979, 19.0402),   # Deak Ferenc ter
    'Copenhagen': (55.6759, 12.5694),   # Radhuspladsen
    'Geneva':     (46.2044,  6.1432),   # Place du Molard
    'Ghent':      (51.0543,  3.7232),   # Korenmarkt
    'Lisbon':     (38.7077, -9.1366),   # Praca do Comercio
    'London':     (51.5079, -0.1281),   # Trafalgar Square
    'Lyon':       (45.7578,  4.8320),   # Place Bellecour
    'Madrid':     (40.4168, -3.7038),   # Puerta del Sol
    'Manchester': (53.4794, -2.2453),   # Albert Square
    'Milan':      (45.4642,  9.1900),   # Piazza del Duomo
    'Munich':     (48.1374, 11.5754),   # Marienplatz
    'Oslo':       (59.9139, 10.7522),   # Karl Johans gate
    'Rome':       (41.8967, 12.4828),   # Piazza Venezia
    'Sicily':     (38.1157, 13.3613),   # Quattro Canti, Palermo
    'Stockholm':  (59.3251, 18.0710),   # Stortorget
    'Vienna':     (48.2082, 16.3738),   # Stephansplatz
    'Zurich':     (47.3700,  8.5400),   # Paradeplatz
}


def haversine_km(lat1, lon1, lat2, lon2):
    # Great-circle distance in kilometres
    r = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def parse_amenities(value):
    if pd.isna(value):
        return 0
    try:
        parsed = ast.literal_eval(value)
        return len(parsed) if isinstance(parsed, list) else 0
    except (ValueError, SyntaxError):
        return 0


def main():
    print('Loading cleaned data...')
    df = pd.read_csv(src_path, low_memory=False)
    print(f'  rows={len(df):,}  cols={len(df.columns)}')

    # ---- distance to city centre ----
    print('\nComputing distance to city centre...')
    centre_lat = df['city'].map(lambda c: CITY_CENTRES.get(c, (np.nan, np.nan))[0])
    centre_lon = df['city'].map(lambda c: CITY_CENTRES.get(c, (np.nan, np.nan))[1])

    unmatched = df['city'][centre_lat.isna()].unique()
    if len(unmatched) > 0:
        print(f'  WARNING: cities without a centre defined: {list(unmatched)}')

    df['distance_to_centre_km'] = haversine_km(
        df['latitude'].astype(float),
        df['longitude'].astype(float),
        centre_lat.astype(float),
        centre_lon.astype(float),
    )
    print(f'  mean={df["distance_to_centre_km"].mean():.2f} km, '
          f'median={df["distance_to_centre_km"].median():.2f} km, '
          f'p99={df["distance_to_centre_km"].quantile(0.99):.2f} km')

    # ---- amenities count ----
    print('\nCounting amenities...')
    df['amenities_count'] = df['amenities'].apply(parse_amenities)
    print(f'  mean={df["amenities_count"].mean():.1f}, '
          f'min={df["amenities_count"].min()}, '
          f'max={df["amenities_count"].max()}')

    # ---- frequency encoding ----
    print('\nFrequency encoding property_type and neighbourhood_cleansed...')
    n = len(df)

    if 'property_type' in df.columns:
        prop_counts = df['property_type'].value_counts(dropna=False)
        df['property_type_freq'] = df['property_type'].map(prop_counts) / n
        print(f'  property_type: {prop_counts.size} unique values')

    if 'neighbourhood_cleansed' in df.columns:
        nbh_counts = df['neighbourhood_cleansed'].value_counts(dropna=False)
        df['neighbourhood_freq'] = df['neighbourhood_cleansed'].map(nbh_counts) / n
        print(f'  neighbourhood_cleansed: {nbh_counts.size} unique values')

    # ---- save ----
    print(f'\nSaving to {out_path}')
    df.to_csv(out_path, index=False)
    print(f'  rows={len(df):,}  cols={len(df.columns)}')

    print('\nCorrelation with price_numeric:')
    new_cols = ['distance_to_centre_km', 'amenities_count',
                'property_type_freq', 'neighbourhood_freq']
    for col in new_cols:
        if col in df.columns:
            corr = df[[col, 'price_numeric']].corr().iloc[0, 1]
            print(f'  {col:25s} r = {corr:+.3f}')


if __name__ == '__main__':
    main()
