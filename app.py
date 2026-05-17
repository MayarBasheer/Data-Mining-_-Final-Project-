import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(
    page_title="Smart Housing Data Mining System",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    data = pd.read_csv("clean_housing_data.csv")
    return data

data = load_data()

if "price_per_m2" not in data.columns:
    data["price_per_m2"] = data["price"] / data["square_meters"]

# =========================
# Helper Functions
# =========================
def get_cluster_label(cluster_id):
    labels = {0: "💚 Affordable Area", 1: "🔴 High-Rent Area", 2: "💛 Family Area"}
    return labels.get(int(cluster_id), f"Cluster {int(cluster_id)}")

def get_anomaly_label(anomaly):
    if anomaly == -1:
        return "⚠️ Unusual Listing"
    return "✅ Normal"

def get_match_stars(score):
    if score < 0.05:
        return "⭐⭐⭐⭐⭐ Perfect Match"
    elif score < 0.1:
        return "⭐⭐⭐⭐ Excellent Match"
    elif score < 0.2:
        return "⭐⭐⭐ Good Match"
    elif score < 0.4:
        return "⭐⭐ Fair Match"
    else:
        return "⭐ Partial Match"

def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# =========================
# Sidebar
# =========================
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        "Home",
        "Search Apartments",
        "Recommendation",
        "Compare Apartments",
        "Dataset Overview",
        "Clustering Analysis",
        "Smart Insights",
        "Association Rules",
        "Anomaly Detection"
    ]
)

# =========================
# Home
# =========================
if page == "Home":
    st.title("🏠 Smart Housing Data Mining System")

    st.write("""
    This is an interactive data mining web application for apartment rental listings.
    The system helps users search apartments, get recommendations, understand apartment groups,
    and detect unusual listings using unsupervised data mining techniques.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(data))
    col2.metric("Average Price", f"${data['price'].mean():.0f}")
    col3.metric("Number of Clusters", int(data["cluster"].nunique()))

    st.markdown("---")

    st.markdown("---")
    st.subheader("📖 How to Use This System")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **🔍 Search Apartments**  
        Filter apartments by price, bedrooms, bathrooms, state, or city.
        Use this when you know exactly what you want.

        **⭐ Recommendation**  
        Enter your budget and preferences.
        The system finds the closest matches for you automatically.

        **🔀 Compare Apartments**  
        Select two apartments and compare them side by side.

        **📊 Dataset Overview**  
        Explore the full dataset with statistics and charts.
        """)

    with col_b:
        st.markdown("""
        **🔵 Clustering Analysis**  
        See how apartments are grouped into 3 market segments.

        **🔗 Association Rules**  
        Discover which amenities tend to appear together in listings.

        **💡 Smart Insights**  
        Visual patterns and apartment group summaries.

        **🚨 Anomaly Detection**  
        View unusual listings that don't follow normal market patterns.
        """)

    st.info("💡 Tip: Start with **Recommendation** if you have a budget in mind, or **Search** if you want exact results.")

# =========================
# Search Apartments
# =========================
elif page == "Search Apartments":
    st.title("🔍 Search Apartments")
    st.write("Use filters to search for apartments that exactly match your needs.")

    col1, col2 = st.columns(2)

    with col1:
        min_price = st.number_input(
            "Min Price ($)",
            min_value=int(data["price"].min()),
            max_value=int(data["price"].max()),
            value=500,
            step=50
        )

    with col2:
        max_price = st.number_input(
            "Max Price ($)",
            min_value=int(data["price"].min()),
            max_value=int(data["price"].max()),
            value=2500,
            step=50
        )

    col3, col4 = st.columns(2)

    with col3:
        bedrooms = st.selectbox("Bedrooms", ["Any"] + sorted(data["bedrooms"].dropna().astype(int).unique().tolist()))

    with col4:
        bathrooms = st.selectbox("Bathrooms", ["Any"] + sorted(data["bathrooms"].dropna().round().astype(int).unique().tolist()))

    col5, col6 = st.columns(2)

    with col5:
        selected_state = st.selectbox("State", ["Any"] + sorted(data["state"].dropna().unique().tolist()))

    with col6:
        search_city = st.text_input("Search by City Name")

    # Amenity filter
    all_amenities_list = sorted(set(
        item.strip()
        for amenities in data["amenities"].dropna()
        for item in str(amenities).split(",")
        if item.strip() and item.strip() != "Unknown"
    ))
    selected_amenity_filter = st.selectbox("Filter by Amenity (optional)", ["Any"] + all_amenities_list)

    filtered = data.copy()
    filtered = filtered[(filtered["price"] >= min_price) & (filtered["price"] <= max_price)]

    if bedrooms != "Any":
        filtered = filtered[filtered["bedrooms"].astype(int) == bedrooms]
    if bathrooms != "Any":
        filtered = filtered[filtered["bathrooms"].round().astype(int) == bathrooms]
    if selected_state != "Any":
        filtered = filtered[filtered["state"] == selected_state]
    if search_city:
        filtered = filtered[filtered["cityname"].astype(str).str.contains(search_city, case=False, na=False)]
    if selected_amenity_filter != "Any":
        filtered = filtered[filtered["amenities"].astype(str).str.contains(selected_amenity_filter, case=False, na=False)]

    st.subheader(f"Search Results: {len(filtered)} apartments found")

    display_filtered = filtered[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "amenities", "pets_allowed", "cluster", "price_per_m2"]].head(100).copy()
    display_filtered["cluster"] = display_filtered["cluster"].apply(lambda x: get_cluster_label(x) if pd.notna(x) else x)
    display_filtered["price_per_m2"] = display_filtered["price_per_m2"].round(2)

    st.dataframe(display_filtered, use_container_width=True)

    # Export button
    st.download_button(
        label="📥 Download Results as CSV",
        data=to_csv(filtered[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "amenities", "pets_allowed", "price_per_m2"]]),
        file_name="search_results.csv",
        mime="text/csv"
    )

# =========================
# Recommendation
# =========================
elif page == "Recommendation":
    st.title("⭐ Apartment Recommendation")

    st.write("""
    Enter your preferences, and the system will recommend apartments closest to your needs.

    This system uses a **Content-based** approach — it calculates a normalized similarity score
    for each apartment. Unlike search, recommendation is flexible and returns similar apartments,
    not only exact matches.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        budget = st.number_input("Your Budget ($)", min_value=100, max_value=10000, value=1500, step=100)
    with col2:
        user_bedrooms = st.number_input("Preferred Bedrooms", min_value=0, max_value=10, value=2)
    with col3:
        user_bathrooms = st.number_input("Preferred Bathrooms", min_value=1, max_value=10, value=1)

    selected_state = st.selectbox("Preferred State (optional)", ["Any"] + sorted(data["state"].dropna().unique().tolist()))

    # Amenity filter
    all_amenities_rec = sorted(set(
        item.strip()
        for amenities in data["amenities"].dropna()
        for item in str(amenities).split(",")
        if item.strip() and item.strip() != "Unknown"
    ))
    selected_amenity_rec = st.selectbox("Preferred Amenity (optional)", ["Any"] + all_amenities_rec)

    top_n = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("Get Recommendations"):
        temp = data.dropna(subset=["price", "bedrooms", "bathrooms", "square_meters", "cluster"]).copy()

        if selected_state != "Any":
            temp = temp[temp["state"] == selected_state]

        if selected_amenity_rec != "Any":
            temp = temp[temp["amenities"].astype(str).str.contains(selected_amenity_rec, case=False, na=False)]

        if len(temp) == 0:
            st.warning(f"No apartments found with '{selected_amenity_rec}' in the selected filters. Try removing the amenity filter or changing the state.")
        else:
            max_bedrooms = max(temp["bedrooms"].max(), 1)
            max_bathrooms = max(temp["bathrooms"].max(), 1)

            temp["score"] = (
                abs(temp["price"] - budget) / budget +
                abs(temp["bedrooms"] - user_bedrooms) / max_bedrooms +
                abs(temp["bathrooms"] - user_bathrooms) / max_bathrooms
            )

            recommendations = temp.sort_values("score").head(top_n)

            st.subheader("Recommended Apartments")
            st.caption("Lower similarity score = better match to your preferences")

            for _, row in recommendations.iterrows():
                st.markdown("---")
                st.markdown(f"### 🏠 Apartment in {row['cityname']}, {row['state']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"${row['price']}")
                c2.metric("Area", f"{row['square_meters']:.1f} m²")
                c3.metric("Bedrooms", int(row["bedrooms"]))
                c4.metric("Bathrooms", int(round(row["bathrooms"])))
                st.write("**Price per m²:**", f"${row['price_per_m2']:.2f}")
                st.write("**Amenities:**", row["amenities"])
                st.write("**Pets Allowed:**", row["pets_allowed"])
                st.write("**Area Type:**", get_cluster_label(row["cluster"]))
                st.write("**Match Quality:**", get_match_stars(row["score"]))

            # Export button
            st.markdown("---")
            st.download_button(
                label="📥 Download Recommendations as CSV",
                data=to_csv(recommendations[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "amenities", "pets_allowed", "price_per_m2"]]),
                file_name="recommendations.csv",
                mime="text/csv"
            )

# =========================
# Compare Apartments
# =========================
elif page == "Compare Apartments":
    st.title("🔀 Compare Apartments")
    st.write("Filter apartments first, then select two to compare side by side.")

    st.subheader("🔍 Filter Apartments")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_state = st.selectbox(
            "State",
            ["Any"] + sorted(data["state"].dropna().unique().tolist()),
            key="compare_state"
        )

    with col2:
        filter_min_price = st.number_input(
            "Min Price ($)",
            min_value=int(data["price"].min()),
            max_value=int(data["price"].max()),
            value=500,
            step=50,
            key="compare_min"
        )

    with col3:
        filter_max_price = st.number_input(
            "Max Price ($)",
            min_value=int(data["price"].min()),
            max_value=int(data["price"].max()),
            value=3000,
            step=50,
            key="compare_max"
        )

    filter_bedrooms = st.selectbox(
        "Bedrooms",
        ["Any"] + sorted(data["bedrooms"].dropna().astype(int).unique().tolist()),
        key="compare_beds"
    )

    # Apply filters
    sample = data.dropna(subset=["price", "bedrooms", "bathrooms", "cityname", "state"]).copy()
    sample = sample[(sample["price"] >= filter_min_price) & (sample["price"] <= filter_max_price)]

    if filter_state != "Any":
        sample = sample[sample["state"] == filter_state]

    if filter_bedrooms != "Any":
        sample = sample[sample["bedrooms"].astype(int) == filter_bedrooms]

    st.write(f"**{len(sample)} apartments match your filters.**")

    if len(sample) < 2:
        st.warning("Not enough apartments match your filters. Please adjust the filters.")
    else:
        sample["label"] = sample.apply(
            lambda r: f"${int(r['price'])} | {int(r['bedrooms'])}bed | {r['cityname']}, {r['state']}",
            axis=1
        )

        options = sample["label"].head(300).tolist()

        st.markdown("---")
        st.subheader("🏠 Select Two Apartments")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Apartment A**")
            choice_a = st.selectbox("Select Apartment A", options, key="a")

        with col2:
            st.markdown("**Apartment B**")
            choice_b = st.selectbox("Select Apartment B", options, index=1, key="b")

        apt_a = sample[sample["label"] == choice_a].iloc[0]
        apt_b = sample[sample["label"] == choice_b].iloc[0]

        st.markdown("---")
        st.subheader("📊 Side by Side Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 🏠 {apt_a['cityname']}, {apt_a['state']}")
            st.metric("Price", f"${apt_a['price']}")
            st.metric("Area", f"{apt_a['square_meters']:.1f} m²")
            st.metric("Bedrooms", int(apt_a["bedrooms"]))
            st.metric("Bathrooms", int(round(apt_a["bathrooms"])))
            st.metric("Price per m²", f"${apt_a['price_per_m2']:.2f}")
            st.write("**Amenities:**", apt_a["amenities"])
            st.write("**Pets Allowed:**", apt_a["pets_allowed"])
            st.write("**Area Type:**", get_cluster_label(apt_a["cluster"]))

        with col2:
            st.markdown(f"### 🏠 {apt_b['cityname']}, {apt_b['state']}")
            st.metric("Price", f"${apt_b['price']}", delta=f"${int(apt_b['price'] - apt_a['price'])}")
            st.metric("Area", f"{apt_b['square_meters']:.1f} m²", delta=f"{apt_b['square_meters'] - apt_a['square_meters']:.1f} m²")
            st.metric("Bedrooms", int(apt_b["bedrooms"]), delta=int(apt_b["bedrooms"] - apt_a["bedrooms"]))
            st.metric("Bathrooms", int(round(apt_b["bathrooms"])), delta=int(round(apt_b["bathrooms"] - apt_a["bathrooms"])))
            st.metric("Price per m²", f"${apt_b['price_per_m2']:.2f}", delta=f"${apt_b['price_per_m2'] - apt_a['price_per_m2']:.2f}")
            st.write("**Amenities:**", apt_b["amenities"])
            st.write("**Pets Allowed:**", apt_b["pets_allowed"])
            st.write("**Area Type:**", get_cluster_label(apt_b["cluster"]))

        # Winner summary
        st.markdown("---")
        st.subheader("🏆 Quick Summary")

        if apt_a["price"] < apt_b["price"]:
            st.success(f"💰 Apartment A is cheaper by ${int(apt_b['price'] - apt_a['price'])}/month")
        elif apt_b["price"] < apt_a["price"]:
            st.success(f"💰 Apartment B is cheaper by ${int(apt_a['price'] - apt_b['price'])}/month")
        else:
            st.info("💰 Both apartments have the same price")

        if apt_a["square_meters"] > apt_b["square_meters"]:
            st.success(f"📐 Apartment A is larger by {apt_a['square_meters'] - apt_b['square_meters']:.1f} m²")
        elif apt_b["square_meters"] > apt_a["square_meters"]:
            st.success(f"📐 Apartment B is larger by {apt_b['square_meters'] - apt_a['square_meters']:.1f} m²")

        if apt_a["price_per_m2"] < apt_b["price_per_m2"]:
            st.success(f"⭐ Apartment A offers better value (${apt_a['price_per_m2']:.2f}/m² vs ${apt_b['price_per_m2']:.2f}/m²)")
        elif apt_b["price_per_m2"] < apt_a["price_per_m2"]:
            st.success(f"⭐ Apartment B offers better value (${apt_b['price_per_m2']:.2f}/m² vs ${apt_a['price_per_m2']:.2f}/m²)")

# =========================
# Dataset Overview
# =========================
elif page == "Dataset Overview":
    st.title("📊 Market Overview")
    st.write("A quick summary of the apartment rental market in our dataset.")

    st.markdown("---")
    st.subheader("🏠 Market at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(data):,}")
    col2.metric("Avg Monthly Rent", f"${data['price'].mean():.0f}")
    col3.metric("Cheapest Listing", f"${int(data['price'].min())}")
    col4.metric("Most Expensive", f"${int(data['price'].max())}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Size", f"{data['square_meters'].mean():.0f} m²")
    col6.metric("Avg Bedrooms", f"{data['bedrooms'].mean():.1f}")
    col7.metric("Avg Price/m²", f"${data['price_per_m2'].mean():.2f}")
    col8.metric("States Covered", data["state"].nunique())

    st.markdown("---")
    st.subheader("🐾 Quick Facts")

    pets_pct = (data["pets_allowed"].astype(str).str.lower().str.contains("cat|dog|yes", na=False).sum() / len(data)) * 100
    photo_pct = (data["has_photo"].astype(str).str.lower().str.contains("yes|thumbnail", na=False).sum() / len(data)) * 100
    anomaly_pct = (data["anomaly"] == -1).sum() / len(data) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("🐾 Pet-Friendly Listings", f"{pets_pct:.0f}%")
    col2.metric("📸 Listings with Photos", f"{photo_pct:.0f}%")
    col3.metric("⚠️ Unusual Listings", f"{anomaly_pct:.0f}%")

    st.markdown("---")
    st.subheader("📍 Most Popular States")

    top_states = data["state"].value_counts().head(5).reset_index()
    top_states.columns = ["State", "Listings"]
    top_states["Avg Price"] = top_states["State"].apply(lambda s: f"${data[data['state'] == s]['price'].mean():.0f}")
    st.dataframe(top_states, use_container_width=True)

    st.markdown("---")
    st.subheader("💰 Price Distribution")
    st.write("Most apartments are priced between $500 and $2,000 per month.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data["price"], bins=50, color="#4C72B0", edgecolor="white")
    ax.axvline(data["price"].mean(), color="#C44E52", linestyle="--", label=f"Avg: ${data['price'].mean():.0f}")
    ax.set_xlabel("Monthly Rent ($)")
    ax.set_ylabel("Number of Listings")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Average Rent by State (Top 8)")

    top8_states = data["state"].value_counts().head(8).index.tolist()
    avg_price_by_state = data[data["state"].isin(top8_states)].groupby("state")["price"].mean().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    avg_price_by_state.plot(kind="bar", ax=ax2, color="#55A868", edgecolor="white")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Average Rent ($)")
    ax2.tick_params(axis="x", rotation=0)
    st.pyplot(fig2)

# =========================
# Clustering Analysis
# =========================
elif page == "Clustering Analysis":
    st.title("🔵 Clustering Analysis")

    st.write("K-Means clustering (k=3) groups similar apartments based on price, area, rooms, bathrooms, location, and price per m².")

    st.subheader("Cluster Distribution")
    cluster_counts = data["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts["Cluster"] = cluster_counts["Cluster"].apply(lambda x: get_cluster_label(x))
    st.dataframe(cluster_counts, use_container_width=True)

    st.subheader("Cluster Summary (Average Values)")
    cluster_summary = data.groupby("cluster")[["price", "square_meters", "bedrooms", "bathrooms", "price_per_m2"]].mean().round(2)
    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Cluster Interpretation")
    st.markdown("""
    - **💚 Cluster 0 — Affordable Area** → Small apartments (~$17/m²), mostly in Texas. Best for individuals on a budget.
    - **🔴 Cluster 1 — High-Rent Area** → Compact apartments (~$33/m²), mostly in California (LA, SF, Seattle). Price driven by location.
    - **💛 Cluster 2 — Family Area** → Large apartments (~$14/m²), mostly in Texas. Best for families needing more space.
    """)

    st.subheader("Filter Apartments by Cluster")
    selected_cluster = st.selectbox("Choose Cluster", sorted(data["cluster"].dropna().unique().astype(int).tolist()), format_func=lambda x: get_cluster_label(x))
    cluster_data = data[data["cluster"] == selected_cluster]
    st.write(f"Number of apartments in {get_cluster_label(selected_cluster)}: {len(cluster_data)}")
    st.dataframe(cluster_data[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "amenities", "price_per_m2"]].head(50), use_container_width=True)

    st.download_button(
        label="📥 Download Cluster Data as CSV",
        data=to_csv(cluster_data[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "amenities", "price_per_m2"]]),
        file_name=f"cluster_{selected_cluster}_data.csv",
        mime="text/csv"
    )

# =========================
# Smart Insights
# =========================
elif page == "Smart Insights":
    st.title("💡 Smart Housing Insights")

    st.write("This page presents user-friendly insights extracted from the data mining process.")

    st.info("Technical note: PCA was used to reduce 7 numerical features into 2 pattern dimensions, preserving 64.4% of the original data information.")

    st.subheader("🏠 Apartment Groups Summary")

    cluster_summary = data.groupby("cluster")[["price", "square_meters", "bedrooms", "bathrooms", "price_per_m2"]].mean().round(2)
    cluster_counts = data["cluster"].value_counts().sort_index()

    cluster_descriptions = {
        0: ("💚 Affordable Small Apartments", "Lowest price with small size. Mostly found in Texas. Best for individuals on a budget.", "success"),
        1: ("🔴 High-Rent Compact Apartments", "Highest price despite small size. Mostly in California (LA, SF, Seattle). Price driven by location.", "error"),
        2: ("💛 Large Family Apartments", "Largest size with most bedrooms. Lowest price per m². Mostly in Texas. Best for families.", "warning")
    }

    for cluster_id in sorted(data["cluster"].dropna().unique()):
        cluster_id = int(cluster_id)
        row = cluster_summary.loc[cluster_id]
        st.markdown("---")
        st.markdown(f"### {get_cluster_label(cluster_id)}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Avg Price", f"${row['price']:.0f}")
        col2.metric("Avg Area", f"{row['square_meters']:.1f} m²")
        col3.metric("Avg Bedrooms", f"{row['bedrooms']:.1f}")
        col4.metric("Price/m²", f"${row['price_per_m2']:.2f}")
        col5.metric("Listings", int(cluster_counts.loc[cluster_id]))
        title, desc, style = cluster_descriptions.get(cluster_id, ("", "", "info"))
        st.markdown(f"**{title}**")
        if style == "success":
            st.success(desc)
        elif style == "error":
            st.error(desc)
        elif style == "warning":
            st.warning(desc)
        else:
            st.info(desc)

    st.subheader("🗺️ Apartment Pattern Map")
    st.write("Each point represents an apartment. Similar apartments appear closer together. Colors represent the three apartment groups.")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {0.0: "#4C72B0", 1.0: "#DD8452", 2.0: "#55A868"}
    labels = {0.0: "💚 Affordable", 1.0: "🔴 High-Rent", 2.0: "💛 Family"}

    for cluster_id, color in colors.items():
        mask = data["cluster"] == cluster_id
        ax.scatter(data.loc[mask, "pca1"], data.loc[mask, "pca2"], c=color, alpha=0.5, label=labels[cluster_id], s=10)

    ax.set_title("Apartment Pattern Map")
    ax.set_xlabel("Pattern Dimension 1")
    ax.set_ylabel("Pattern Dimension 2")
    ax.legend()
    st.pyplot(fig)
    st.caption("This map is created using PCA. It helps visualize apartment groups without requiring technical knowledge.")

# =========================
# Association Rules
# =========================
elif page == "Association Rules":
    st.title("🔗 Association Rule Mining")

    @st.cache_data
    def compute_rules_full():
        rules_data = data[["price", "square_meters", "pets_allowed", "has_photo", "amenities"]].copy()

        rules_data["price_category"] = pd.cut(
            rules_data["price"],
            bins=[0, 949, 1270, 1695, 100000],
            labels=["low", "medium", "high", "very_high"]
        )

        rules_data["size_category"] = pd.cut(
            rules_data["square_meters"],
            bins=[0, 60, 75, 102, 100000],
            labels=["small", "medium", "large", "very_large"]
        )

        def split_amenities(x):
            if pd.isna(x) or x == "Unknown":
                return []
            return [item.strip().lower() for item in str(x).split(",") if item.strip()]

        transactions = []
        for _, row in rules_data.iterrows():
            items = []
            items.append("price_" + str(row["price_category"]))
            items.append("size_" + str(row["size_category"]))
            items.append("pets_" + str(row["pets_allowed"]).lower())
            items.append("photo_" + str(row["has_photo"]).lower())
            amenities_list = split_amenities(row["amenities"])
            items.extend(["amenity_" + a for a in amenities_list[:10]])
            transactions.append(items)

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
        rules = rules.sort_values(by="lift", ascending=False)

        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

        return rules

    with st.spinner("Computing association rules... this may take a moment."):
        all_rules = compute_rules_full()

    tab1, tab2 = st.tabs(["🏠 Amenity Explorer", "📊 Technical View"])

    with tab1:
        st.subheader("Discover What Comes With Your Amenity")
        st.write("Select an amenity you're interested in, and we'll show you what other features are commonly found in the same apartments.")

        all_amenities = sorted(set(
            item.replace("amenity_", "").title()
            for item in all_rules["antecedents_str"].str.split(", ").explode()
            if item.startswith("amenity_")
        ))

        if all_amenities:
            selected_amenity = st.selectbox("Select an amenity you want:", all_amenities)
            amenity_key = "amenity_" + selected_amenity.lower()

            matching_rules = all_rules[
                all_rules["antecedents_str"].str.contains(amenity_key, na=False)
            ].head(5)

            if len(matching_rules) > 0:
                st.markdown(f"### 🏠 Apartments with **{selected_amenity}** usually also have:")
                for _, rule in matching_rules.iterrows():
                    consequents = rule["consequents_str"]
                    confidence = rule["confidence"]
                    items = [
                        item.replace("amenity_", "").replace("pets_", "Pets: ").replace("photo_", "Photo: ").title()
                        for item in consequents.split(", ")
                    ]
                    for item in items:
                        st.markdown(f"✅ **{item}** — found together {confidence*100:.0f}% of the time")
            else:
                st.info(f"No strong rules found for {selected_amenity}. Try another amenity.")

        st.markdown("---")
        st.success("""
        💡 **Key Insight**: Luxury amenities come as a package!
        If you find an apartment with Patio/Deck and Internet Access,
        it's very likely to also have Parking, Dishwasher, and Cable TV.
        """)

    with tab2:
        st.subheader("Association Rules — Technical Details")
        st.info("""
        - **Support**: How often these features appear together in the dataset
        - **Confidence**: If feature A exists, probability that feature B also exists
        - **Lift > 1**: The relationship is real, not a coincidence
        """)
        display_rules = all_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(20).copy()
        display_rules.columns = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
        st.dataframe(display_rules.round(4), use_container_width=True)

        st.subheader("🔑 Key Finding")
        st.success("""
        Luxury amenities tend to appear together as a package.
        Apartments with patio/deck, internet access, parking, and dishwasher
        are very likely to also have cable TV, pet-friendly policies, and garbage disposal.
        Lift values above 9.0 confirm these are strong, non-random relationships.
        """)

# =========================
# Anomaly Detection
# =========================
elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    st.write("""
    Isolation Forest is used to detect unusual apartment listings.
    These apartments may have unusual prices, sizes, or feature combinations
    that don't follow normal market patterns.
    """)

    st.info("""
    How it works: The algorithm isolates each data point by randomly splitting the data.
    Unusual listings are easier to isolate and receive a shorter path,
    marking them as anomalies (3% of total listings).
    """)

    anomalies = data[data["anomaly"] == -1]
    normal = data[data["anomaly"] == 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", len(data))
    col2.metric("Normal Listings", len(normal))
    col3.metric("Anomalies Detected", len(anomalies))

    st.subheader("Unusual Apartment Listings")
    st.write("""
    Common characteristics of detected anomalies:
    - Very small size (less than 25 m²)
    - Zero bedrooms with high price
    - Unusually high price per m² compared to market average
    """)

    st.dataframe(
        anomalies[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "price_per_m2", "amenities"]].head(100),
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Anomalies as CSV",
        data=to_csv(anomalies[["price", "square_meters", "bedrooms", "bathrooms", "cityname", "state", "price_per_m2", "amenities"]]),
        file_name="anomalies.csv",
        mime="text/csv"
    )

    st.subheader("Anomalies vs Normal: Price Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal["price"], bins=50, alpha=0.6, label="✅ Normal", color="#55A868")
    ax.hist(anomalies["price"], bins=30, alpha=0.8, label="⚠️ Anomalies", color="#C44E52")
    ax.set_title("Price Distribution: Normal vs Anomalies")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)