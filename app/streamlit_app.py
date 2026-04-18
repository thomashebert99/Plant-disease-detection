"""Streamlit frontend for the plant disease diagnosis API."""

from __future__ import annotations

import base64
import html
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import streamlit as st
from dotenv import load_dotenv
from app.disease_info import DISEASE_INFO, DISEASE_LABELS, SPECIES_LABELS

DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 60

SPECIES_OPTIONS = SPECIES_LABELS


@dataclass(slots=True)
class ApiResponse:
    """HTTP response returned by the API client."""

    status_code: int
    payload: dict[str, Any]


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

    render_last_result()


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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_api_url() -> str:
    """Return the API base URL from the environment."""

    return os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


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
            st.image(uploaded_file, caption="Image chargée", width=420)
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


def render_monitoring_page(api_url: str) -> None:
    """Render a lightweight monitoring view for API predictions."""

    st.subheader("Monitoring du service")
    st.write(
        "Cette page résume les prédictions traitées par l'API. "
        "Elle ne stocke pas les images envoyées."
    )

    if st.button("Rafraîchir", use_container_width=False):
        cached_get_monitoring_summary.clear()

    response = get_monitoring_summary(api_url)
    if response.status_code != 200:
        st.error(response.payload.get("detail", "Monitoring indisponible."))
        return

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

    second_row = st.columns(3, gap="medium")
    second_row[0].metric("Latence moyenne", format_ms(payload.get("average_latency_ms")))
    second_row[1].metric(
        "Confiance espèce",
        format_optional_percent(payload.get("average_species_confidence")),
    )
    second_row[2].metric(
        "Confiance maladie",
        format_optional_percent(payload.get("average_disease_confidence")),
    )

    last_event_at = payload.get("last_event_at")
    if last_event_at:
        st.caption(f"Dernier événement : {last_event_at}")
    else:
        st.info("Aucune prédiction enregistrée depuis le dernier démarrage de l'API.")

    with st.expander("Réponse brute de l'API"):
        st.json(payload)


def render_last_result() -> None:
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
        return

    render_successful_prediction(payload)


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

    gradcam_base64 = payload.get("gradcam_base64")
    if gradcam_base64:
        st.image(
            base64.b64decode(gradcam_base64),
            caption="Zones analysées par le modèle",
            width=480,
        )

    with st.expander("Détails techniques"):
        st.json(payload)


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
            st.image(image_bytes, caption=image_name, width=340)

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
            <p><strong>Agent causatif :</strong> {html.escape(str(info["agent"]))}</p>
            <p><strong>Traitement curatif :</strong> {html.escape(str(info["curative"]))}</p>
            <p><strong>Traitement préventif :</strong> {html.escape(str(info["preventive"]))}</p>
            <p><strong>Saison / gravité :</strong> {html.escape(str(info["season_gravity"]))}</p>
        </div>
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


def call_predict_api(
    *,
    api_url: str,
    image_bytes: bytes,
    filename: str,
    species: str | None,
) -> ApiResponse:
    """Call `POST /predict` with multipart form data."""

    fields = {"species": species} if species else {}
    body, content_type = build_multipart_body(
        fields=fields,
        file_field="file",
        filename=filename,
        file_bytes=image_bytes,
    )
    http_request = request.Request(
        f"{api_url}/predict",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    return send_json_request(http_request, timeout=REQUEST_TIMEOUT_SECONDS)


def get_api_health(api_url: str) -> ApiResponse:
    """Call `GET /health`."""

    return cached_get_api_health(api_url)


def get_models_info(api_url: str) -> ApiResponse:
    """Call `GET /models/info`."""

    return cached_get_models_info(api_url)


def get_monitoring_summary(api_url: str) -> ApiResponse:
    """Call `GET /monitoring/summary`."""

    return cached_get_monitoring_summary(api_url)


@st.cache_data(ttl=10, show_spinner=False)
def cached_get_api_health(api_url: str) -> ApiResponse:
    """Call `GET /health` with a short cache to keep the UI responsive."""

    http_request = request.Request(f"{api_url}/health", method="GET")
    return send_json_request(http_request, timeout=3)


@st.cache_data(ttl=10, show_spinner=False)
def cached_get_models_info(api_url: str) -> ApiResponse:
    """Call `GET /models/info` with a short cache to avoid repeated polling."""

    http_request = request.Request(f"{api_url}/models/info", method="GET")
    return send_json_request(http_request, timeout=5)


@st.cache_data(ttl=5, show_spinner=False)
def cached_get_monitoring_summary(api_url: str) -> ApiResponse:
    """Call `GET /monitoring/summary` with a short cache for the dashboard."""

    http_request = request.Request(f"{api_url}/monitoring/summary", method="GET")
    return send_json_request(http_request, timeout=5)


def send_json_request(http_request: request.Request, *, timeout: int) -> ApiResponse:
    """Send an HTTP request and decode a JSON response."""

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = decode_json_body(response.read())
            return ApiResponse(status_code=response.status, payload=payload)
    except error.HTTPError as exc:
        payload = decode_json_body(exc.read())
        return ApiResponse(status_code=exc.code, payload=payload)
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return ApiResponse(
            status_code=0,
            payload={"detail": f"Impossible de joindre le service : {exc}"},
        )


def decode_json_body(body: bytes) -> dict[str, Any]:
    """Decode an HTTP body as JSON with a readable fallback for proxy errors."""

    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        compact_text = " ".join(text.split())
        detail = compact_text[:500] if compact_text else "Réponse non JSON vide."
        return {"detail": detail}

    if isinstance(payload, dict):
        return payload
    return {"detail": str(payload)}


def build_multipart_body(
    *,
    fields: dict[str, str | None],
    file_field: str,
    filename: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    """Build a multipart/form-data request body without an extra dependency."""

    boundary = f"----plant-disease-detection-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def label_to_species(label: str) -> str:
    """Return the API species key matching a French display label."""

    reverse_mapping = {display: key for key, display in SPECIES_OPTIONS.items()}
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
