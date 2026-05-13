import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from inference import (
    predict_mri,
    predict_ct,
    predict_fusion,
    apply_heatmap_on_image,
    compute_validation_metrics
)

def run_system(mri_file, ct_file):

    if mri_file is None or ct_file is None:
        return "Upload MRI and CT images.", None, None, None, None

    # ================= SINGLE IMAGE PREDICTION =================
    mri_pred, mri_conf, probs, mri_img = predict_mri(mri_file.name)
    ct_pred, ct_conf, ct_probs, ct_img = predict_ct(ct_file.name)
    fusion_pred, fusion_conf, fusion_probs, fused_img = predict_fusion(
        mri_file.name, ct_file.name
    )

    # ================= VALIDATION METRICS =================
    val_metrics = compute_validation_metrics()

    result_text = (
        "========== SECTION 1: SINGLE IMAGE PREDICTION ==========\n\n"
        f"Prediction (MRI Model): {mri_pred}\n"
        f"Confidence: {mri_conf:.4f}\n\n"
        f"Prediction (CT Model): {ct_pred}\n"
        f"Confidence: {ct_conf:.4f}\n\n"
         "========== SECTION 1: FUSED IMAGE PREDICTION ==========\n\n"

        "Fusion Model Result:\n"
        f"Prediction: {fusion_pred}\n"
        f"Confidence: {fusion_conf:.4f}\n"
        f"Agreement with MRI: {'YES' if mri_pred == fusion_pred else 'NO'}\n"
        f"Agreement with CT: {'YES' if ct_pred == fusion_pred else 'NO'}\n"
        f"Confidence Gain (Fusion - MRI): {fusion_conf - mri_conf:.4f}\n"
        f"Confidence Gain (Fusion - CT): {fusion_conf - ct_conf:.4f}\n\n"

        "========== SECTION 2: MODEL VALIDATION PERFORMANCE ==========\n\n"
        f"Accuracy: {val_metrics['accuracy']*100:.2f}%\n"
        f"Sensitivity: {val_metrics['sensitivity']*100:.2f}%\n"
        f"Specificity: {val_metrics['specificity']*100:.2f}%\n"
        f"F1-Score: {val_metrics['f1_score']*100:.2f}%\n"
        f"AUC: {val_metrics['auc']:.3f}"
    )

    # ================= HEATMAPS =================
    mri_heatmap = apply_heatmap_on_image(mri_img)
    ct_heatmap = apply_heatmap_on_image(ct_img)
    fusion_heatmap = apply_heatmap_on_image(fused_img / 255.0)

    # ================= PROBABILITY CHART =================
    prob_fig = plt.figure()
    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    plt.bar(class_names, fusion_probs)
    plt.title("Fusion Model Class Probabilities")
    plt.ylabel("Probability")
    plt.xticks(rotation=20)
    plt.tight_layout()

    # ================= CONFIDENCE COMPARISON CHART =================
    confidence_fig = plt.figure()
    models = ["MRI Model", "CT Model", "Fusion Model"]
    confidences = [mri_conf, ct_conf, fusion_conf]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plt.bar(models, confidences, color=colors)
    plt.ylim(0, 1)
    plt.title("MRI vs CT vs Fusion Confidence Comparison")
    plt.ylabel("Confidence")
    for index, value in enumerate(confidences):
        plt.text(index, value + 0.02, f"{value:.2f}", ha='center')
    plt.tight_layout()

    return result_text, mri_heatmap, ct_heatmap, fusion_heatmap, fused_img, prob_fig, confidence_fig


with gr.Blocks() as demo:

    gr.Markdown("# 🧠 MRI vs MRI+CT Tumor Comparison System")

    with gr.Row():
        mri_input = gr.File(label="Upload MRI DICOM")
        ct_input = gr.File(label="Upload CT DICOM")

    run_btn = gr.Button("Analyze")

    result_output = gr.Textbox(label="Prediction & Validation Summary", lines=25)
    mri_map = gr.Image(label="MRI Heatmap")
    ct_map = gr.Image(label="CT Heatmap")
    fusion_map = gr.Image(label="Fusion Heatmap")
    fused_img_out = gr.Image(label="Fused MRI + CT Image")
    prob_chart = gr.Plot(label="Class Probability Chart")
    confidence_chart = gr.Plot(label="MRI vs CT vs Fusion Confidence")

    run_btn.click(
        run_system,
        inputs=[mri_input, ct_input],
        outputs=[result_output, mri_map, ct_map, fusion_map, fused_img_out, prob_chart, confidence_chart]
    )

demo.launch()