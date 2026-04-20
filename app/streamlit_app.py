"""Streamlit frontend for the plant disease diagnosis API."""

from __future__ import annotations

import base64
import html
from typing import Any

import streamlit as st
from dotenv import load_dotenv

try:
    from app.api_client import (
        cached_get_monitoring_events,
        cached_get_monitoring_summary,
        call_predict_api,
        get_api_health,
        get_api_url,
        get_models_info,
        get_monitoring_events,
        get_monitoring_summary,
        submit_feedback,
    )
except ModuleNotFoundError:
    from api_client import (
        cached_get_monitoring_events,
        cached_get_monitoring_summary,
        call_predict_api,
        get_api_health,
        get_api_url,
        get_models_info,
        get_monitoring_events,
        get_monitoring_summary,
        submit_feedback,
    )

try:
    from app.disease_info import DISEASE_INFO, DISEASE_LABELS, SPECIES_LABELS
except ModuleNotFoundError:
    from disease_info import DISEASE_INFO, DISEASE_LABELS, SPECIES_LABELS

SPECIES_OPTIONS = SPECIES_LABELS


def main() -> None:
    """Render the Streamlit application."""

    load_dotenv()
    configure_page()

    api_url = get_api_url()
    render_header()
    page = render_sidebar(api_url)

    if page == "Monitoring":
        render_monitoring_page(api_url)
        return

    progress_placeholder = st.empty()
    uploaded_file, selected_species, analyze_clicked = render_input_panel()

    if analyze_clicked and uploaded_file is not None:
        run_prediction(
            api_url=api_url,
            uploaded_file=uploaded_file,
            species=selected_species,
            progress_placeholder=progress_placeholder,
        )

    render_last_result(api_url)


def configure_page() -> None:
    """Configure page metadata and styling."""

    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1120px;
            padding-top: 2rem;
        }
        .diagnosis-badge {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            min-height: 130px;
        }
        .diagnosis-label {
            color: #465264;
            font-size: 0.86rem;
            margin-bottom: 0.25rem;
        }
        .diagnosis-value {
            color: #18212f;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .confidence-bar {
            height: 4px;
            background: #e2e8f0;
            border-radius: 2px;
            margin-top: 0.55rem;
            margin-bottom: 0.35rem;
        }
        .confidence-bar.compact {
            height: 5px;
            margin-top: 0.25rem;
            margin-bottom: 0.6rem;
        }
        .confidence-bar-fill {
            height: 100%;
            background: #16735f;
            border-radius: 2px;
        }
        .diagnosis-confidence {
            color: #16735f;
            font-size: 0.86rem;
        }
        .responsive-image {
            width: min(100%, 360px);
            max-height: 360px;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .responsive-image.upload-preview {
            width: min(100%, 420px);
            max-height: 320px;
        }
        .image-caption {
            color: #6b7280;
            font-size: 0.84rem;
            text-align: center;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }
        .info-panel {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: #f8fafc;
            margin-top: 0.8rem;
        }
        .info-panel h4 {
            margin-top: 0;
            margin-bottom: 0.75rem;
            color: #18212f;
        }
        .info-panel p {
            margin: 0.45rem 0;
            line-height: 1.45;
        }
        .info-row {
            display: grid;
            grid-template-columns: 1.45rem minmax(0, 1fr);
            column-gap: 0.35rem;
            align-items: start;
            margin: 0.6rem 0;
            line-height: 1.45;
        }
        .info-icon {
            text-align: center;
            line-height: 1.45;
        }
        .ranked-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-top: 0.4rem;
        }
        .ranked-label {
            font-weight: 650;
            color: #18212f;
        }
        .ranked-confidence {
            color: #16735f;
            font-weight: 700;
        }
        @media (prefers-color-scheme: dark) {
            .diagnosis-badge {
                background: #1a2332;
                border-color: #2d4060;
            }
            .diagnosis-label { color: #8ba4c0; }
            .diagnosis-value { color: #e2eaf4; }
            .confidence-bar { background: #2d4060; }
            .confidence-bar-fill { background: #4dba99; }
            .diagnosis-confidence { color: #4dba99; }
            .info-panel {
                background: #111827;
                border-color: #2d4060;
            }
            .info-panel h4,
            .ranked-label { color: #e2eaf4; }
            .ranked-confidence { color: #4dba99; }
            .image-caption { color: #9ca3af; }
        }
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .responsive-image,
            .responsive-image.upload-preview {
                width: min(100%, 320px);
                max-height: 280px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the page title and short description."""

    st.title("🌿 Diagnostic foliaire par image")
    st.caption(
        "Chargez une photo de feuille pour identifier l'espèce et détecter une maladie éventuelle."
    )


def render_sidebar(api_url: str) -> str:
    """Render service status and supported species."""

    st.sidebar.title("Plant Disease Detection")
    st.sidebar.caption("Diagnostic de maladies foliaires par IA")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=["Diagnostic", "Monitoring"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    health = get_api_health(api_url)
    models_info = get_models_info(api_url)

    st.sidebar.subheader("État du service")
    if health.status_code == 200:
        st.sidebar.success("Service en ligne")
    else:
        st.sidebar.error("Service hors ligne")

    if models_info.status_code == 200 and models_info.payload.get("config_available"):
        st.sidebar.success("Modèles chargés")
    elif models_info.status_code == 200:
        st.sidebar.warning("Modèles en attente")

    st.sidebar.divider()
    st.sidebar.subheader("Espèces supportées")
    for label in SPECIES_OPTIONS.values():
        st.sidebar.markdown(f"• {label}")

    return page


def render_input_panel() -> tuple[Any | None, str | None, bool]:
    """Render upload, analysis mode controls and submit button."""

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Photo de feuille",
            type=["jpg", "jpeg", "png", "webp"],
            help="Utilisez une image nette, idéalement centrée sur une feuille.",
        )
        if uploaded_file is not None:
            render_responsive_image(
                uploaded_file.getvalue(),
                caption="Image chargée",
                css_class="upload-preview",
            )
        else:
            st.info("Ajoutez une image pour lancer le diagnostic.")

    with right:
        st.subheader("Mode d'analyse")
        mode = st.radio(
            "Choisir le mode",
            ["Automatique", "Manuel"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if mode == "Automatique":
            st.write("L'espèce est détectée automatiquement avant le diagnostic maladie.")
        else:
            st.write("Indiquez l'espèce pour un diagnostic direct, sans détection préalable.")

        selected_species = None
        if mode == "Manuel":
            selected_label = st.selectbox(
                "Espèce connue",
                options=list(SPECIES_OPTIONS.values()),
                index=0,
            )
            selected_species = label_to_species(selected_label)

        st.write("")
        analyze_clicked = st.button(
            "Analyser l'image",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

    return uploaded_file, selected_species, analyze_clicked


def run_prediction(
    *,
    api_url: str,
    uploaded_file: Any,
    species: str | None,
    progress_placeholder: Any,
) -> None:
    """Call the prediction API and store the latest result in session state."""

    image_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name or "leaf.png"

    progress_placeholder.progress(
        10,
        text="Préparation de l'image. Le premier appel peut prendre un peu de temps.",
    )
    progress_placeholder.progress(35, text="Envoi à l'API et chargement des modèles.")
    response = call_predict_api(
        api_url=api_url,
        image_bytes=image_bytes,
        filename=filename,
        species=species,
    )
    progress_placeholder.progress(100, text="Analyse terminée.")
    progress_placeholder.empty()

    st.session_state["last_response"] = response.payload
    st.session_state["last_status_code"] = response.status_code
    st.session_state["last_image_bytes"] = image_bytes
    st.session_state["last_image_name"] = filename
    st.session_state["last_feedback_sent"] = False


def render_monitoring_page(api_url: str) -> None:
    """Render a visual monitoring dashboard for API predictions."""

    st.subheader("Monitoring du modèle et du service")
    st.write(
        "Cette page suit la santé de l'API, la confiance du modèle, les signaux de drift "
        "et les retours utilisateur. Les images envoyées ne sont pas conservées."
    )

    if st.button("Rafraîchir", use_container_width=False):
        cached_get_monitoring_summary.clear()
        cached_get_monitoring_events.clear()

    response = get_monitoring_summary(api_url)
    if response.status_code != 200:
        st.error(response.payload.get("detail", "Monitoring indisponible."))
        return
    events_response = get_monitoring_events(api_url, limit=100)

    payload = response.payload
    total_predictions = int(payload.get("total_predictions") or 0)
    ok_count = int(payload.get("ok") or 0)
    uncertain_count = int(payload.get("uncertain_species") or 0)
    error_count = int(payload.get("errors") or 0)

    first_row = st.columns(4, gap="medium")
    first_row[0].metric("Prédictions", total_predictions)
    first_row[1].metric("Réponses OK", ok_count)
    first_row[2].metric("Espèces incertaines", uncertain_count)
    first_row[3].metric("Erreurs", error_count)

    second_row = st.columns(4, gap="medium")
    second_row[0].metric("Latence moyenne", format_ms(payload.get("average_latency_ms")))
    second_row[1].metric("Latence P95", format_ms(payload.get("p95_latency_ms")))
    second_row[2].metric(
        "Confiance espèce",
        format_optional_percent(payload.get("average_species_confidence")),
    )
    second_row[3].metric(
        "Confiance maladie",
        format_optional_percent(payload.get("average_disease_confidence")),
    )

    render_alerts(payload.get("alerts", []))

    st.divider()
    render_distribution_section(payload)

    st.divider()
    render_confidence_section(payload)

    st.divider()
    render_drift_section(
        payload.get("domain_shift", {}),
        payload.get("model_quality_shift", {}),
    )

    st.divider()
    render_feedback_summary(payload.get("feedback", {}))

    if events_response.status_code == 200:
        st.divider()
        render_recent_events(events_response.payload.get("events", []))
    else:
        st.warning(events_response.payload.get("detail", "Événements indisponibles."))

    last_event_at = payload.get("last_event_at")
    if last_event_at:
        st.caption(f"Dernier événement : {last_event_at}")
    else:
        st.info("Aucune prédiction enregistrée depuis le dernier démarrage de l'API.")

    with st.expander("Réponse brute de l'API"):
        st.json(payload)


def render_alerts(alerts: Any) -> None:
    """Render active monitoring alerts."""

    st.subheader("Alertes actives")
    if not alerts:
        st.success("Aucune alerte active sur la fenêtre observée.")
        return
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        message = alert.get("message", "Alerte monitoring")
        value = alert.get("value")
        threshold = alert.get("threshold")
        st.warning(f"{message} Valeur : {value} ; seuil : {threshold}.")


def render_distribution_section(payload: dict[str, Any]) -> None:
    """Render predicted class distributions."""

    st.subheader("Distribution des prédictions")
    left, right, third = st.columns(3, gap="medium")
    with left:
        render_bar_chart("Espèces prédites", payload.get("species_distribution", {}))
    with right:
        render_bar_chart("Maladies prédites", payload.get("disease_distribution", {}))
    with third:
        healthy_ratio = payload.get("healthy_ratio")
        st.metric("Ratio feuilles saines", format_optional_percent(healthy_ratio))
        st.caption("Calculé uniquement sur les diagnostics maladie disponibles.")


def render_confidence_section(payload: dict[str, Any]) -> None:
    """Render confidence histograms."""

    st.subheader("Fiabilité des scores")
    left, right = st.columns(2, gap="medium")
    with left:
        render_bar_chart("Confiance espèce", payload.get("species_confidence_histogram", {}))
    with right:
        render_bar_chart("Confiance maladie", payload.get("disease_confidence_histogram", {}))


def render_drift_section(domain_shift: Any, model_quality_shift: Any) -> None:
    """Render the domain-shift summary."""

    st.subheader("Détection du drift")
    if not isinstance(domain_shift, dict) or not domain_shift.get("reference_available"):
        st.info("Référence de drift indisponible. Le monitoring reste limité aux métriques service.")
        return

    status = str(domain_shift.get("status", "insufficient_data"))
    risk_level = str(domain_shift.get("risk_level", "none"))
    closest_reference = domain_shift.get("closest_reference") or "non déterminée"
    status_label = {
        "in_domain": "In-domain",
        "ood_like": "OOD connu",
        "reference_shift": "Décalage",
        "unknown_shift": "Inconnu",
        "insufficient_data": "Attente",
    }.get(status, status)

    cols = st.columns(4, gap="medium")
    cols[0].metric("État domaine", status_label)
    cols[1].metric("Risque", risk_level)
    cols[2].metric("Référence proche", str(closest_reference))
    cols[3].metric("Fenêtre", int(domain_shift.get("window_size") or 0))
    st.caption(
        "OOD connu signifie que le flux ressemble à la référence PlantDoc : "
        "la fiabilité est surveillée, sans conclure automatiquement à une erreur."
    )

    distances = domain_shift.get("distances", {})
    render_bar_chart("Distance aux références", distances)

    prediction_drift = domain_shift.get("prediction_drift", {})
    if isinstance(prediction_drift, dict):
        left, right = st.columns(2, gap="medium")
        left.metric("Drift espèces", format_distance(prediction_drift.get("species_distance")))
        right.metric("Drift maladies", format_distance(prediction_drift.get("disease_distance")))

    signals = domain_shift.get("signals", [])
    if signals:
        st.write("Signaux principaux")
        st.dataframe(signals, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun signal fort sur les métriques dérivées de l'image.")

    if isinstance(model_quality_shift, dict):
        st.write("Signal feedback")
        feedback_cols = st.columns(4, gap="medium")
        feedback_cols[0].metric(
            "État qualité",
            str(model_quality_shift.get("status", "n/a")),
        )
        feedback_cols[1].metric(
            "Risque qualité",
            str(model_quality_shift.get("risk_level", "none")),
        )
        feedback_cols[2].metric(
            "Désaccord",
            format_optional_percent(model_quality_shift.get("disagreement_rate")),
        )
        feedback_cols[3].metric(
            "Retours requis",
            int(model_quality_shift.get("minimum_feedback") or 0),
        )
        st.caption(
            "Le feedback ne détecte pas le data drift à lui seul : il apporte une vérité "
            "terrain utilisateur pour signaler une possible dérive de qualité."
        )


def render_feedback_summary(feedback: Any) -> None:
    """Render aggregate user feedback."""

    st.subheader("Retours utilisateur")
    if not isinstance(feedback, dict):
        st.info("Aucun retour utilisateur disponible.")
        return
    cols = st.columns(3, gap="medium")
    cols[0].metric("Retours", int(feedback.get("total_feedback") or 0))
    cols[1].metric("Désaccord", format_optional_percent(feedback.get("disagreement_rate")))
    cols[2].metric("Dernier retour", feedback.get("last_feedback_at") or "Aucun")
    render_bar_chart("Verdicts", feedback.get("verdict_distribution", {}))


def render_recent_events(events: Any) -> None:
    """Render recent prediction events and latency trend."""

    st.subheader("Événements récents")
    if not isinstance(events, list) or not events:
        st.info("Aucun événement récent à afficher.")
        return

    latency_points = [
        {"timestamp": event.get("timestamp", ""), "latency_ms": event.get("latency_ms")}
        for event in events
        if isinstance(event, dict) and isinstance(event.get("latency_ms"), int | float)
    ]
    if latency_points:
        st.line_chart(latency_points, x="timestamp", y="latency_ms")

    visible_events = [
        {
            "timestamp": event.get("timestamp"),
            "status": event.get("status"),
            "species": event.get("species"),
            "disease": event.get("disease"),
            "species_confidence": event.get("species_confidence"),
            "disease_confidence": event.get("disease_confidence"),
            "latency_ms": event.get("latency_ms"),
        }
        for event in events
        if isinstance(event, dict)
    ]
    st.dataframe(visible_events, use_container_width=True, hide_index=True)


def render_bar_chart(title: str, values: Any) -> None:
    """Render a Streamlit bar chart from a mapping."""

    st.write(title)
    if not isinstance(values, dict) or not values:
        st.caption("Aucune donnée.")
        return
    rows = [{"label": str(label), "value": value} for label, value in values.items()]
    st.bar_chart(rows, x="label", y="value")


def format_distance(value: Any) -> str:
    """Format a drift distance value."""

    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.2f}"


def render_last_result(api_url: str) -> None:
    """Render the last API response, if any."""

    payload = st.session_state.get("last_response")
    status_code = st.session_state.get("last_status_code")
    if payload is None:
        return

    st.divider()
    st.subheader("Résultat")

    if status_code != 200:
        detail = payload.get("detail", "Erreur inconnue.")
        if status_code == 503:
            st.warning(detail)
            st.info("Les modèles ne sont pas encore disponibles. Réessayez dans quelques instants.")
        elif status_code == 400:
            st.error(detail)
        else:
            st.error(f"Erreur {status_code} : {detail}")
        return

    if payload.get("status") == "uncertain_species":
        render_uncertain_species(payload)
        render_feedback_form(api_url, payload)
        return

    render_successful_prediction(payload)
    render_feedback_form(api_url, payload)


def render_uncertain_species(payload: dict[str, Any]) -> None:
    """Render the case where the API asks for species confirmation."""

    species = payload.get("species", {})
    st.warning(payload.get("action_required", "Espèce incertaine."))
    render_prediction_overview(
        species=species,
        disease=None,
        include_disease_info=False,
    )
    render_ranked_predictions(
        "Top prédictions espèce",
        species.get("top_predictions", []),
        label_formatter=display_species,
    )
    st.info("Passez en mode **Manuel** avec l'espèce correcte pour lancer le diagnostic maladie.")


def render_successful_prediction(payload: dict[str, Any]) -> None:
    """Render a successful diagnosis."""

    species = payload.get("species", {})
    disease = payload.get("disease") or {}
    render_prediction_overview(
        species=species,
        disease=disease,
        include_disease_info=True,
    )

    render_ranked_predictions(
        "Top prédictions espèce",
        species.get("top_predictions", []),
        label_formatter=display_species,
    )
    render_ranked_predictions(
        "Top prédictions maladie",
        disease.get("top_predictions", []),
        label_formatter=lambda label: display_disease(label, species.get("species")),
    )

    with st.expander("Détails techniques"):
        st.json(payload)


def render_feedback_form(api_url: str, payload: dict[str, Any]) -> None:
    """Render a compact user feedback form for iterative monitoring."""

    st.divider()
    st.subheader("Retour sur la prédiction")
    st.caption(
        "Le retour est enregistré pour le monitoring. L'image envoyée n'est pas conservée."
    )

    if st.session_state.get("last_feedback_sent"):
        st.success("Retour déjà enregistré pour cette prédiction.")
        return

    species = payload.get("species", {})
    disease = payload.get("disease") or {}
    verdict_labels = {
        "Correcte": "correct",
        "Incorrecte": "incorrect",
        "Je ne sais pas": "unsure",
    }

    with st.form("prediction_feedback_form", clear_on_submit=False):
        verdict_label = st.radio(
            "Cette prédiction vous semble-t-elle correcte ?",
            options=list(verdict_labels),
            horizontal=True,
        )
        corrected_species = None
        corrected_disease = None
        if verdict_labels[verdict_label] == "incorrect":
            selected_species = st.selectbox(
                "Espèce correcte",
                options=["Non renseigné", *SPECIES_OPTIONS.values()],
            )
            selected_disease = st.selectbox(
                "Maladie correcte",
                options=["Non renseignée", *DISEASE_LABELS.values()],
            )
            corrected_species = (
                label_to_species(selected_species)
                if selected_species != "Non renseigné"
                else None
            )
            corrected_disease = (
                label_to_disease(selected_disease)
                if selected_disease != "Non renseignée"
                else None
            )
        comment = st.text_area("Commentaire optionnel", max_chars=500)
        submitted = st.form_submit_button("Envoyer le retour")

    if not submitted:
        return

    feedback_payload = {
        "verdict": verdict_labels[verdict_label],
        "prediction_status": payload.get("status"),
        "predicted_species": species.get("species"),
        "predicted_disease": disease.get("disease"),
        "corrected_species": corrected_species,
        "corrected_disease": corrected_disease,
        "comment": comment.strip() or None,
    }
    response = submit_feedback(api_url, feedback_payload)
    if response.status_code == 200 and response.payload.get("stored"):
        st.session_state["last_feedback_sent"] = True
        cached_get_monitoring_summary.clear()
        st.success(response.payload.get("message", "Retour enregistré."))
    else:
        st.error(response.payload.get("detail", "Impossible d'enregistrer le retour."))


def render_prediction_overview(
    *,
    species: dict[str, Any],
    disease: dict[str, Any] | None,
    include_disease_info: bool,
) -> None:
    """Render the main result with the uploaded image and disease information."""

    left, right = st.columns([0.8, 1.6], gap="large")

    with left:
        image_bytes = st.session_state.get("last_image_bytes")
        image_name = st.session_state.get("last_image_name", "Image envoyée")
        if image_bytes:
            render_responsive_image(image_bytes, caption=image_name)

    with right:
        disease_label = disease.get("disease") if disease else None
        species_label = species.get("species")
        render_badges(
            [
                ("Espèce", display_species(species_label), species.get("confidence")),
                (
                    "Maladie",
                    display_disease(disease_label, species_label)
                    if disease_label
                    else "En attente de confirmation",
                    disease.get("confidence") if disease else None,
                ),
            ]
        )
        if include_disease_info:
            render_disease_information(species_label, disease_label)


def render_disease_information(species: Any, disease: Any) -> None:
    """Render the agronomic information card for the predicted disease."""

    if not disease or disease == "Healthy":
        st.success("Aucune maladie foliaire détectée parmi les classes entraînées.")
        return

    info = DISEASE_INFO.get((str(species), str(disease)))
    if info is None:
        st.info("Informations détaillées indisponibles pour cette classe.")
        return

    title = html.escape(str(info["title"]))
    english = html.escape(str(info["english"]))
    url = html.escape(str(info["url"]))

    st.markdown(f"### À propos - [{title}]({url}) (*{english}*)")
    st.markdown(
        f"""
        <div class="info-panel">
            <h4>Plus d'informations</h4>
            <div class="info-row">
                <span class="info-icon">🌿</span>
                <span><strong>Agent causatif :</strong> {html.escape(str(info["agent"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">🩺</span>
                <span><strong>Traitement curatif :</strong> {html.escape(str(info["curative"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">💊</span>
                <span><strong>Traitement préventif :</strong> {html.escape(str(info["preventive"]))}</span>
            </div>
            <div class="info-row">
                <span class="info-icon">🛡️</span>
                <span><strong>Saison / gravité :</strong> {html.escape(str(info["season_gravity"]))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_responsive_image(
    image_bytes: bytes,
    *,
    caption: str,
    css_class: str = "",
) -> None:
    """Render an image with CSS constraints instead of a fixed pixel width."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_caption = html.escape(caption)
    extra_class = f" {css_class}" if css_class else ""
    st.markdown(
        f"""
        <img class="responsive-image{extra_class}" src="data:image/jpeg;base64,{encoded}" alt="{safe_caption}">
        <div class="image-caption">{safe_caption}</div>
        """,
        unsafe_allow_html=True,
    )


def render_ranked_predictions(
    title: str,
    candidates: list[dict[str, Any]],
    *,
    label_formatter: Any,
) -> None:
    """Render the top prediction candidates returned by the API."""

    if not candidates:
        return

    st.markdown(f"#### {title}")
    for rank, candidate in enumerate(candidates[:3], start=1):
        label = label_formatter(candidate.get("label"))
        confidence = float(candidate.get("confidence") or 0.0)
        st.markdown(
            f"""
            <div class="ranked-row">
                <div class="ranked-label">{rank}. {html.escape(label)}</div>
                <div class="ranked-confidence">{confidence:.0%}</div>
            </div>
            <div class="confidence-bar compact">
                <div class="confidence-bar-fill" style="width:{int(confidence * 100)}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_badges(items: list[tuple[str, str, float | None]]) -> None:
    """Render compact result badges with a confidence progress bar."""

    columns = st.columns(len(items), gap="medium")
    for column, (label, value, confidence) in zip(columns, items):
        pct = int(float(confidence) * 100) if confidence is not None else 0
        safe_label = html.escape(label)
        safe_value = html.escape(value)
        safe_confidence = html.escape(format_confidence(confidence))
        bar_html = (
            f'<div class="confidence-bar">'
            f'<div class="confidence-bar-fill" style="width:{pct}%"></div>'
            f"</div>"
            if confidence is not None
            else ""
        )
        with column:
            st.markdown(
                f"""
                <div class="diagnosis-badge">
                    <div class="diagnosis-label">{safe_label}</div>
                    <div class="diagnosis-value">{safe_value}</div>
                    {bar_html}
                    <div class="diagnosis-confidence">{safe_confidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def label_to_species(label: str) -> str:
    """Return the API species key matching a French display label."""

    reverse_mapping = {display: key for key, display in SPECIES_OPTIONS.items()}
    return reverse_mapping[label]


def label_to_disease(label: str) -> str:
    """Return the API disease key matching a French display label."""

    reverse_mapping = {display: key for key, display in DISEASE_LABELS.items()}
    return reverse_mapping[label]


def display_species(value: Any) -> str:
    """Format a species key for display."""

    if not value:
        return "Non déterminée"
    return SPECIES_OPTIONS.get(str(value), str(value))


def display_disease(value: Any, species: Any | None = None) -> str:
    """Format a disease label for display."""

    if not value:
        return "Non déterminée"
    if species and (str(species), str(value)) in DISEASE_INFO:
        return str(DISEASE_INFO[(str(species), str(value))]["title"])
    return DISEASE_LABELS.get(str(value), str(value).replace("_", " "))


def format_confidence(value: float | None) -> str:
    """Format confidence as a percentage."""

    if value is None:
        return "Confiance non disponible"
    return f"Confiance : {float(value):.0%}"


def format_optional_percent(value: Any) -> str:
    """Format a nullable API ratio as a compact percentage."""

    if value is None:
        return "N/A"
    return f"{float(value):.0%}"


def format_ms(value: Any) -> str:
    """Format a nullable latency value."""

    if value is None:
        return "N/A"
    return f"{float(value):.0f} ms"


if __name__ == "__main__":
    main()
