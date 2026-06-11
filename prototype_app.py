import os
import json
import math
import pandas as pd
import joblib
import streamlit as st

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Models', 'xgboost_tweedie.joblib')
lookup_path = os.path.join(script_dir, 'Models', 'prototype_lookups.json')

CITY_CENTRES = {
    'Amsterdam':  (52.3731,  4.8926),
    'Berlin':     (52.5163, 13.3777),
    'Brussels':   (50.8467,  4.3525),
    'Budapest':   (47.4979, 19.0402),
    'Copenhagen': (55.6759, 12.5694),
    'Geneva':     (46.2044,  6.1432),
    'Ghent':      (51.0543,  3.7232),
    'Lisbon':     (38.7077, -9.1366),
    'London':     (51.5079, -0.1281),
    'Lyon':       (45.7578,  4.8320),
    'Madrid':     (40.4168, -3.7038),
    'Manchester': (53.4794, -2.2453),
    'Milan':      (45.4642,  9.1900),
    'Munich':     (48.1374, 11.5754),
    'Oslo':       (59.9139, 10.7522),
    'Rome':       (41.8967, 12.4828),
    'Sicily':     (38.1157, 13.3613),
    'Stockholm':  (59.3251, 18.0710),
    'Vienna':     (48.2082, 16.3738),
    'Zurich':     (47.3700,  8.5400),
}


@st.cache_resource
def load_model():
    return joblib.load(model_path)


@st.cache_data
def load_lookups():
    with open(lookup_path, 'r') as f:
        return json.load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = lat2r - lat1r
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


st.set_page_config(page_title='Airbnb Price Estimator', layout='centered')
st.title('🏠 Airbnb Nightly Price Estimator')
st.caption('XGBoost (Tweedie loss) trained on ~297k listings across 20 European cities.')

model = load_model()
lookups = load_lookups()

with st.sidebar:
    st.header('Listing details')
    city = st.selectbox('City', lookups['cities'], index=lookups['cities'].index('London'))

    centre_lat, centre_lon = CITY_CENTRES[city]
    distance_km = st.slider('Distance from city centre (km)', 0.0, 50.0, 2.0, step=0.5)

    room_type = st.selectbox(
        'Room type',
        ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'],
    )
    property_type = st.selectbox(
        'Property type', lookups['property_types_top20'], index=0,
    )
    nbh_options = lookups['neighbourhoods_by_city'].get(city, [])
    neighbourhood = st.selectbox(
        'Neighbourhood (top 15 in city)',
        nbh_options if nbh_options else ['(unknown)'],
    )

    st.markdown('---')
    accommodates = st.number_input('Guests', 1, 16, 2)
    bedrooms = st.number_input('Bedrooms', 0, 10, 1)
    beds = st.number_input('Beds', 0, 16, 1)
    bathrooms = st.number_input('Bathrooms', 0.0, 10.0, 1.0, step=0.5)
    amenities_count = st.slider('Amenities count', 0, 100, 30)
    minimum_nights = st.number_input('Minimum nights', 1, 365, 2)

    st.markdown('---')
    review_rating = st.slider('Review score (rating)', 0.0, 5.0, 4.7, 0.1)
    review_clean = st.slider('Review score (cleanliness)', 0.0, 5.0, 4.7, 0.1)
    review_loc = st.slider('Review score (location)', 0.0, 5.0, 4.7, 0.1)
    review_value = st.slider('Review score (value)', 0.0, 5.0, 4.6, 0.1)
    number_of_reviews = st.number_input('Number of reviews', 0, 2000, 50)

    st.markdown('---')
    superhost = st.checkbox('Superhost', value=False)
    instant = st.checkbox('Instant bookable', value=True)
    host_listings = st.number_input('Host listings count', 1, 500, 1)
    availability_365 = st.slider('Availability (days/year)', 0, 365, 200)


bearing_offset_lat = distance_km / 111.0
listing_lat = centre_lat + bearing_offset_lat * 0.7
listing_lon = centre_lon + (bearing_offset_lat * 0.7) / max(math.cos(math.radians(centre_lat)), 0.1)

prop_freq = lookups['property_type_freq'].get(property_type, 0.001)
nbh_freq = lookups['neighbourhood_freq'].get(neighbourhood, 0.001)

input_df = pd.DataFrame([{
    'accommodates': accommodates,
    'bedrooms': bedrooms,
    'beds': beds,
    'bathrooms': bathrooms,
    'latitude': listing_lat,
    'longitude': listing_lon,
    'minimum_nights': minimum_nights,
    'number_of_reviews': number_of_reviews,
    'review_scores_rating': review_rating,
    'review_scores_cleanliness': review_clean,
    'review_scores_location': review_loc,
    'review_scores_value': review_value,
    'calculated_host_listings_count': host_listings,
    'availability_365': availability_365,
    'distance_to_centre_km': distance_km,
    'amenities_count': amenities_count,
    'property_type_freq': prop_freq,
    'neighbourhood_freq': nbh_freq,
    'room_type': room_type,
    'city': city,
    'host_is_superhost': 't' if superhost else 'f',
    'instant_bookable': 't' if instant else 'f',
}])


pred_price = float(max(model.predict(input_df)[0], 1.0))

low = pred_price * 0.7
high = pred_price * 1.3

col1, col2, col3 = st.columns(3)
col1.metric('Estimate', f'€{pred_price:,.0f}')
col2.metric('Lower band (≈ -30%)', f'€{low:,.0f}')
col3.metric('Upper band (≈ +30%)', f'€{high:,.0f}')

st.subheader('Inputs summary')
st.dataframe(input_df.T.rename(columns={0: 'value'}), use_container_width=True)

with st.expander('Notes'):
    st.markdown(
        'The estimate comes from an XGBoost regressor with Tweedie loss '
        '(R² ≈ 0.60, MAE ≈ €44 on a held-out test set). Tweedie regression '
        'directly models a right-skewed positive target, which behaves '
        'better on luxury listings than a log-transformed MSE objective. '
        'The ±30% band is a rough guide rather than a formal prediction '
        'interval; on bookings above €500/night the underlying error can '
        'still be considerably larger. Trained on September 2025 Inside '
        'Airbnb snapshots from 20 European cities.'
    )
