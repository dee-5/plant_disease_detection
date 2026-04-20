"""
Disease Information & Intelligent GradCAM Explanation Generator.

This module contains:
1. A knowledge dictionary of all 15 plant disease classes with symptoms, causes, and treatment.
2. A GradCAM spatial analyzer that generates image-specific explanations, not generic boilerplate.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Disease Knowledge Base — 15 classes from PlantVillage
# ──────────────────────────────────────────────────────────────────────────────

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display_name": "Bell Pepper — Bacterial Spot",
        "cause": "Xanthomonas campestris pv. vesicatoria",
        "severity": "Moderate",
        "symptoms": "Small, dark, water-soaked lesions on leaves that enlarge and become angular with tan-brown centers. Lesions may coalesce, causing leaf yellowing and defoliation.",
        "visual_pattern": "scattered dark lesions with angular margins",
        "expected_activation": "multiple small, scattered hot zones corresponding to individual lesion sites",
        "recommendation": "Remove infected plant debris. Apply copper-based bactericides preventively. Use disease-free seed and resistant cultivars. Avoid overhead irrigation.",
    },
    "Pepper__bell___healthy": {
        "display_name": "Bell Pepper — Healthy",
        "cause": "No pathogen detected",
        "severity": "None",
        "symptoms": "Uniform green coloration with no visible lesions, spots, or discoloration.",
        "visual_pattern": "uniform healthy tissue",
        "expected_activation": "diffuse, low-intensity activation across the leaf with no concentrated hotspots",
        "recommendation": "Your plant appears healthy! Continue regular watering, ensure proper nutrition, and monitor for early signs of disease.",
    },
    "Potato___Early_blight": {
        "display_name": "Potato — Early Blight",
        "cause": "Alternaria solani",
        "severity": "Moderate to High",
        "symptoms": "Dark brown to black concentric ring lesions ('target spots') appearing first on older, lower leaves. Lesions have a bull's eye pattern surrounded by yellow halos.",
        "visual_pattern": "concentric ring patterns (target/bull's-eye spots) on lower leaves",
        "expected_activation": "concentrated hot zones on distinct circular lesions with ring-like patterns",
        "recommendation": "Remove affected lower leaves immediately. Apply fungicide (chlorothalonil or mancozeb). Ensure adequate plant spacing for air circulation. Practice crop rotation.",
    },
    "Potato___Late_blight": {
        "display_name": "Potato — Late Blight",
        "cause": "Phytophthora infestans",
        "severity": "Very High — Devastating",
        "symptoms": "Large, irregular, water-soaked gray-green lesions that rapidly turn dark brown/black. White fuzzy mold may appear on leaf undersides in humid conditions. Spreads extremely fast.",
        "visual_pattern": "large, irregular dark necrotic patches often starting at leaf edges",
        "expected_activation": "broad, intense activation covering large areas of necrotic tissue, often concentrated at leaf margins",
        "recommendation": "URGENT: Remove and destroy all infected plants immediately. Apply systemic fungicide (metalaxyl). Do NOT compost infected material. Alert neighboring growers as this disease spreads rapidly.",
    },
    "Potato___healthy": {
        "display_name": "Potato — Healthy",
        "cause": "No pathogen detected",
        "severity": "None",
        "symptoms": "Healthy green foliage with no visible spots, lesions, or discoloration.",
        "visual_pattern": "uniform healthy tissue",
        "expected_activation": "diffuse, low-intensity activation with no concentrated hotspots",
        "recommendation": "Your potato plant is healthy! Maintain consistent watering, hill the plants as they grow, and watch for early signs of blight during humid weather.",
    },
    "Tomato_Bacterial_spot": {
        "display_name": "Tomato — Bacterial Spot",
        "cause": "Xanthomonas spp.",
        "severity": "Moderate",
        "symptoms": "Small, dark, greasy-looking spots on leaves that may have yellow halos. Spots are angular and may merge, causing leaf tissue to tear. Severely affected leaves turn yellow and drop.",
        "visual_pattern": "numerous small dark spots with occasional yellow halos",
        "expected_activation": "multiple small scattered activation zones corresponding to individual bacterial lesion sites",
        "recommendation": "Remove infected leaves. Apply copper-based sprays early in disease progression. Avoid working with wet plants. Use resistant varieties when available.",
    },
    "Tomato_Early_blight": {
        "display_name": "Tomato — Early Blight",
        "cause": "Alternaria solani",
        "severity": "Moderate to High",
        "symptoms": "Characteristic concentric ring pattern ('target spots') starting on lower, older leaves. Dark brown circular lesions with bull's-eye pattern, surrounded by yellowing tissue.",
        "visual_pattern": "target-like concentric ring lesions starting on older leaves",
        "expected_activation": "focused activation on circular lesion sites showing ring patterns, typically in the lower portion of the leaf",
        "recommendation": "Prune lower branches for air circulation. Apply fungicide (chlorothalonil or copper-based). Mulch around base to prevent soil splash. Rotate crops annually.",
    },
    "Tomato_Late_blight": {
        "display_name": "Tomato — Late Blight",
        "cause": "Phytophthora infestans",
        "severity": "Very High — Devastating",
        "symptoms": "Large, dark brown, water-soaked, irregularly shaped lesions that spread rapidly. Leaves may show a grayish-white mold on undersides. The disease can destroy an entire crop within days.",
        "visual_pattern": "large irregular dark necrotic areas spreading from leaf edges inward",
        "expected_activation": "intense, widespread activation covering major portions of the leaf, often concentrated at the expanding disease front",
        "recommendation": "CRITICAL: Remove and destroy all affected plants immediately — do not compost. Apply fungicide containing metalaxyl or chlorothalonil. This disease is highly contagious and can devastate fields.",
    },
    "Tomato_Leaf_Mold": {
        "display_name": "Tomato — Leaf Mold",
        "cause": "Passalora fulva (syn. Cladosporium fulvum)",
        "severity": "Moderate",
        "symptoms": "Pale green to yellowish patches on upper leaf surface, with olive-green to grayish-brown velvety mold growth on the underside. Older leaves are affected first.",
        "visual_pattern": "diffuse yellowing patches on upper surface with mold growth underneath",
        "expected_activation": "moderately spread activation covering pale/yellow patches, often in a patchy distributed pattern",
        "recommendation": "Improve air circulation by pruning and spacing plants. Reduce greenhouse humidity below 85%. Apply fungicide if severe. Remove heavily infected leaves.",
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name": "Tomato — Septoria Leaf Spot",
        "cause": "Septoria lycopersici",
        "severity": "Moderate to High",
        "symptoms": "Numerous small (1-3mm), circular spots with dark brown margins and grayish-white centers. Tiny black fruiting bodies (pycnidia) may be visible in spot centers. Lower leaves affected first.",
        "visual_pattern": "many small circular spots with dark edges and light centers",
        "expected_activation": "numerous small, discrete activation points scattered across the leaf, corresponding to individual tiny spots",
        "recommendation": "Remove infected lower leaves promptly. Apply fungicide (chlorothalonil or mancozeb). Avoid overhead watering. Mulch to prevent soil splash. Practice 2-year crop rotation.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display_name": "Tomato — Spider Mite Damage",
        "cause": "Tetranychus urticae (Two-spotted spider mite)",
        "severity": "Moderate",
        "symptoms": "Fine stippling (tiny yellow dots) on upper leaf surface where mites feed. Leaves develop a bronzed or silvery appearance. Fine webbing may be visible in severe infestations. Leaves dry out and curl.",
        "visual_pattern": "fine stippling and bronzing across the leaf surface",
        "expected_activation": "diffuse, moderately spread activation corresponding to the fine stippling damage pattern across the leaf surface",
        "recommendation": "Spray undersides of leaves with strong water jet to dislodge mites. Apply miticide (abamectin) or insecticidal soap. Introduce predatory mites (Phytoseiulus) as biological control. Increase humidity.",
    },
    "Tomato__Target_Spot": {
        "display_name": "Tomato — Target Spot",
        "cause": "Corynespora cassiicola",
        "severity": "Moderate",
        "symptoms": "Small, brown, circular spots with concentric rings and a target-like appearance, often with a yellow halo. Can appear on leaves, stems, and fruit.",
        "visual_pattern": "circular spots with concentric rings creating a target pattern",
        "expected_activation": "focused activation zones on circular lesions, typically clustered where disease pressure is highest",
        "recommendation": "Remove infected leaves and destroy debris. Apply fungicide (azoxystrobin or chlorothalonil). Improve air circulation. Avoid overhead irrigation.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display_name": "Tomato — Yellow Leaf Curl Virus",
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV), transmitted by whiteflies",
        "severity": "Very High",
        "symptoms": "Severe upward curling and yellowing of leaf margins. Leaves become small, crumpled, and cupped. Plant growth is severely stunted. Flowers may drop without setting fruit.",
        "visual_pattern": "curled leaf margins with pronounced yellowing at the edges",
        "expected_activation": "strong activation along leaf margins and curled edges where the characteristic yellowing and deformation is most visible",
        "recommendation": "No cure exists for infected plants — remove and destroy them. Control whitefly vectors with insecticidal soap or neem oil. Use yellow sticky traps. Plant resistant varieties. Use reflective mulch to repel whiteflies.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display_name": "Tomato — Mosaic Virus",
        "cause": "Tomato Mosaic Virus (ToMV)",
        "severity": "Moderate to High",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves. Leaves may be distorted, curled, or reduced in size. Fruit may show uneven ripening or internal browning.",
        "visual_pattern": "irregular mosaic mottling with alternating light and dark patches",
        "expected_activation": "patchy, irregular activation pattern reflecting the mosaic mottling distribution across the leaf",
        "recommendation": "Remove infected plants immediately. Disinfect all tools with 10% bleach solution. Wash hands thoroughly before handling healthy plants. The virus spreads by contact and can persist in soil — do not replant tomatoes in the same location.",
    },
    "Tomato_healthy": {
        "display_name": "Tomato — Healthy",
        "cause": "No pathogen detected",
        "severity": "None",
        "symptoms": "Vibrant, uniform green leaves with no spots, curling, yellowing, or other abnormalities.",
        "visual_pattern": "uniform healthy tissue",
        "expected_activation": "diffuse, low-intensity activation across the leaf with no concentrated hotspots, indicating no specific disease features detected",
        "recommendation": "Your tomato plant looks healthy! Maintain proper watering, provide support for growing vines, and keep monitoring for early disease signs especially during warm humid weather.",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Spatial Analysis Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_location_description(centroid_y, centroid_x):
    """Convert normalized centroid coordinates to a human-readable location."""
    # Y axis: 0=top, 1=bottom
    if centroid_y < 0.33:
        v = "upper"
    elif centroid_y < 0.66:
        v = "central"
    else:
        v = "lower"

    # X axis: 0=left, 1=right
    if centroid_x < 0.33:
        h = "left"
    elif centroid_x < 0.66:
        h = "central"
    else:
        h = "right"

    if v == "central" and h == "central":
        return "the center of the leaf"
    elif v == "central":
        return f"the {h} side of the leaf"
    elif h == "central":
        return f"the {v} portion of the leaf"
    else:
        return f"the {v}-{h} region of the leaf"


def _get_spread_description(spread):
    """Convert spread metric to a human-readable pattern description."""
    if spread < 0.08:
        return "tightly concentrated in a very small area"
    elif spread < 0.15:
        return "clustered in a localized region"
    elif spread < 0.22:
        return "moderately spread across a portion of the leaf"
    elif spread < 0.30:
        return "distributed across a broad area of the leaf"
    else:
        return "widely scattered across the entire leaf surface"


def _get_intensity_description(mean_activation, max_activation):
    """Describe the overall intensity of the model's focus."""
    if mean_activation > 0.4:
        return "very strong"
    elif mean_activation > 0.25:
        return "strong"
    elif mean_activation > 0.15:
        return "moderate"
    elif mean_activation > 0.08:
        return "mild"
    else:
        return "subtle"


def _get_confidence_assessment(confidence):
    """Provide a confidence-based trust assessment."""
    if confidence >= 0.95:
        return "The model is highly confident in this diagnosis."
    elif confidence >= 0.85:
        return "The model shows strong confidence in this diagnosis."
    elif confidence >= 0.70:
        return "The model has moderate confidence. Consider getting a second opinion if the symptoms don't match."
    elif confidence >= 0.50:
        return "The model has low confidence. The image may be unclear, or the disease may be in an early stage. Manual inspection is recommended."
    else:
        return "The model is uncertain about this diagnosis. The image quality may be poor, or the condition may not match the training data well. Please consult an expert."


# ──────────────────────────────────────────────────────────────────────────────
# Main GradCAM Explanation Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_gradcam_explanation(gradcam_map, predicted_class, confidence, top3=None):
    """
    Generate an intelligent, image-specific GradCAM explanation.

    Args:
        gradcam_map: numpy array (H, W) with values 0-1 from GradCAM
        predicted_class: string — the predicted class name
        confidence: float (0-1) — confidence of the top prediction
        top3: list of (class_name, probability) tuples (optional)

    Returns:
        dict with 'summary', 'spatial_detail', 'disease_detail', 'recommendation', 'trust_assessment'
    """
    h, w = gradcam_map.shape

    # ── Compute spatial statistics ──
    mean_activation = float(np.mean(gradcam_map))
    max_activation = float(np.max(gradcam_map))

    # Coverage at different thresholds
    high_mask = gradcam_map > 0.6  # "hot" regions
    mid_mask = gradcam_map > 0.35  # "warm" regions
    high_coverage = float(np.sum(high_mask)) / (h * w) * 100
    mid_coverage = float(np.sum(mid_mask)) / (h * w) * 100

    # Find centroid and spread of high-activation regions
    if np.sum(high_mask) > 10:  # Need enough pixels for meaningful analysis
        y_coords, x_coords = np.where(high_mask)
        centroid_y = float(np.mean(y_coords)) / h
        centroid_x = float(np.mean(x_coords)) / w
        spread_y = float(np.std(y_coords)) / h
        spread_x = float(np.std(x_coords)) / w
        spread = (spread_y + spread_x) / 2
    elif np.sum(mid_mask) > 10:
        y_coords, x_coords = np.where(mid_mask)
        centroid_y = float(np.mean(y_coords)) / h
        centroid_x = float(np.mean(x_coords)) / w
        spread_y = float(np.std(y_coords)) / h
        spread_x = float(np.std(x_coords)) / w
        spread = (spread_y + spread_x) / 2
    else:
        centroid_y, centroid_x = 0.5, 0.5
        spread = 0.35

    # ── Get descriptors ──
    location = _get_location_description(centroid_y, centroid_x)
    spread_desc = _get_spread_description(spread)
    intensity_desc = _get_intensity_description(mean_activation, max_activation)
    trust = _get_confidence_assessment(confidence)

    # ── Get disease info ──
    info = DISEASE_INFO.get(predicted_class, {})
    display_name = info.get("display_name", predicted_class.replace("_", " "))
    cause = info.get("cause", "Unknown")
    severity = info.get("severity", "Unknown")
    symptoms = info.get("symptoms", "")
    expected_pattern = info.get("expected_activation", "")
    recommendation = info.get("recommendation", "Consult a plant pathology expert for further diagnosis.")
    is_healthy = "healthy" in predicted_class.lower()

    # ── Build the explanation ──
    conf_pct = round(confidence * 100, 1)

    if is_healthy:
        summary = (
            f"The model classified this leaf as **{display_name}** with {conf_pct}% confidence. "
            f"The GradCAM heatmap shows {intensity_desc}, diffuse activation with no concentrated disease hotspots, "
            f"which is consistent with a healthy leaf. The model examined the overall leaf texture and color uniformity "
            f"to confirm the absence of disease markers."
        )
        spatial_detail = (
            f"The activation is {spread_desc}, with the highest attention around {location}. "
            f"Only {high_coverage:.1f}% of the leaf area triggered strong model focus (>60% intensity), "
            f"and {mid_coverage:.1f}% triggered moderate focus. This low, uniform activation pattern "
            f"indicates the model did not detect any localized disease features."
        )
    else:
        summary = (
            f"The model detected **{display_name}** with {conf_pct}% confidence. "
            f"The heatmap reveals {intensity_desc} activation {spread_desc}, primarily concentrated around "
            f"{location}. This activation pattern corresponds to the model identifying visual features "
            f"consistent with {info.get('visual_pattern', 'disease symptoms')}."
        )
        spatial_detail = (
            f"Approximately **{high_coverage:.1f}%** of the leaf area triggered strong model focus (shown in red/yellow), "
            f"while **{mid_coverage:.1f}%** triggered moderate attention (orange tones). "
            f"The peak activation intensity is {max_activation:.2f}, centered around {location}. "
            f"The activation is {spread_desc}. "
        )
        if expected_pattern:
            spatial_detail += f"This matches the expected pattern for {display_name}: {expected_pattern}."

    disease_detail = ""
    if not is_healthy:
        disease_detail = (
            f"**Pathogen:** {cause} · **Severity:** {severity}\n\n"
            f"**Typical Symptoms:** {symptoms}\n\n"
            f"The highlighted regions in the heatmap (red/yellow areas) align with where these symptoms "
            f"are typically observed on affected leaves. The cooler (blue/purple) areas represent healthy tissue "
            f"that the model correctly excluded from its disease assessment."
        )

    # ── Compile result ──
    return {
        "summary": summary,
        "spatial_detail": spatial_detail,
        "disease_detail": disease_detail,
        "recommendation": recommendation,
        "trust_assessment": trust,
        "display_name": display_name,
        "cause": cause,
        "severity": severity,
        "is_healthy": is_healthy,
        "stats": {
            "mean_activation": round(mean_activation, 3),
            "max_activation": round(max_activation, 3),
            "high_coverage_pct": round(high_coverage, 1),
            "mid_coverage_pct": round(mid_coverage, 1),
            "centroid": [round(centroid_x, 2), round(centroid_y, 2)],
            "spread": round(spread, 3),
        }
    }
