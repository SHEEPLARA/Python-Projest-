
import streamlit as st
import predictor

st.title("Champions League Predictor (xG-based)")

st.write(
    "Predict match outcomes using Monte Carlo simulations. "
    "You can either auto-fetch recent xG from FotMob (team ID), "
    "or enter your own average xG values."
)

mode = st.radio("Data source", ["FotMob (auto)", "Manual xG input"])

st.markdown("---")

if mode == "FotMob (auto)":
    home_id = st.text_input("Home Team FotMob ID", "9823")   # Arsenal
    away_id = st.text_input("Away Team FotMob ID", "9825")   # Bayern
    num_matches = st.slider("Number of past matches to analyse", 3, 10, 5)
else:
    home_name = st.text_input("Home team name", "Arsenal")
    away_name = st.text_input("Away team name", "Bayern Munich")
    hxGF = st.number_input("Home avg xG (last N matches)", value=1.6, step=0.05)
    hxGA = st.number_input("Home avg xGA (last N matches)", value=1.2, step=0.05)
    axGF = st.number_input("Away avg xG (last N matches)", value=1.8, step=0.05)
    axGA = st.number_input("Away avg xGA (last N matches)", value=1.1, step=0.05)

sims = st.slider("Simulations", 5000, 50000, 20000)

if st.button("Predict"):
    try:
        if mode == "FotMob (auto)":
            # Use FotMob IDs + auto xG
            hxGF, hxGA = predictor.get_team_recent_xg(home_id, matches=num_matches)
            axGF, axGA = predictor.get_team_recent_xg(away_id, matches=num_matches)
            home_name = f"Team {home_id}"
            away_name = f"Team {away_id}"

        # Run prediction
        result = predictor.predict_match(
            home_name, away_name, hxGF, hxGA, axGF, axGA, sims=sims
        )

        st.subheader("Results")
        st.write(f"**Fixture:** {result['home_team']} vs {result['away_team']}")
        st.write(f"Expected goals (xG): **{result['home_xg']:.2f} – {result['away_xg']:.2f}**")
        st.write(f"Most likely score: **{result['most_likely_score']}**")

        st.write("### Win/Draw/Loss probabilities")
        st.bar_chart(
            {
                "Probability": [
                    result["home_win"],
                    result["draw"],
                    result["away_win"],
                ]
            }
        )
        st.write(
            f"- Home win: **{result['home_win']*100:.1f}%**  \n"
            f"- Draw: **{result['draw']*100:.1f}%**  \n"
            f"- Away win: **{result['away_win']*100:.1f}%**"
        )

    except Exception as e:
        st.error(str(e))
        st.info(
            "If FotMob auto-fetch keeps failing, switch to 'Manual xG input' "
            "and type in average xG values yourself."
        )
