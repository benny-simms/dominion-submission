import React, { useState, useEffect } from "react";

function Report({ id, modelVersion}) {
    const [report, setReport] = useState(null);
    const [modelMetrics, setModelMetrics] = useState(null);

    useEffect(() => {
        fetch(`http://localhost:8000/reports/${id}`)
            .then(res => res.json())
            .then(setReport)
            .catch(err => console.error("Report fetch error:", err));
    }, [id, modelVersion]);

    // Model‑wide metrics (metrics.json)
    useEffect(() => {
        fetch(`http://localhost:8000/model_metrics`)
            .then(res => res.json())
            .then(setModelMetrics)
            .catch(err => console.error("Model metrics fetch error:", err));
    }, [modelVersion]);   // ← only depends on dataset toggle

    if (!report) {
        return (
            <div style={{color: "var(--text-muted)", padding: 12}}>
                Loading report…
            </div>
        );
    }

    const recall = report.recall.toFixed(2);
    const precision = report.precision.toFixed(2);
    const precision_large = report.precision_large.toFixed(2);
    const precision_small = report.precision_small.toFixed(2);

    return (
        <div style={{marginTop: 20}}>
            <h3
                style={{
                    fontSize: 13,
                    textTransform: "uppercase",
                    letterSpacing: 2,
                    marginBottom: 12
                }}
            >
                Image Performance Metrics:
            </h3>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 8,
                    marginBottom: 12
                }}
            >
                <div>
                    <p style={{color: "var(--text-muted)", fontSize: 11}}>Total Ground Truth</p>
                    <p style={{fontSize: 14}}>{report.n_ground_truth}</p>
                </div>
                <div>
                    <p style={{color: "var(--text-muted)", fontSize: 11}}>Detections</p>
                    <p style={{fontSize: 14}}>{report.n_detections}</p>
                </div>
            </div>

            <div
                style={{
                    borderTop: "1px solid var(--border)",
                    paddingTop: 10,
                    marginTop: 6
                }}
            >
                <h4
                    style={{
                        fontSize: 11,
                        textTransform: "uppercase",
                        letterSpacing: 2,
                        color: "var(--text-muted)",
                        marginBottom: 8
                    }}
                >
                    Classification Accuracy
                </h4>

                <p>True Positives: {report.true_positives}</p>
                <p>False Positives: {report.false_positives}</p>
                <p>False Negatives: {report.false_negatives}</p>
            </div>

            <div
                style={{
                    borderTop: "1px solid var(--border)",
                    paddingTop: 10,
                    marginTop: 10
                }}
            >
                <h4
                    style={{
                        fontSize: 11,
                        textTransform: "uppercase",
                        letterSpacing: 2,
                        color: "var(--text-muted)",
                        marginBottom: 8
                    }}
                >
                    Performance Metrics
                </h4>

                <p>Precision:</p>
                <p>
                    &nbsp; Overall:{" "}
                    <span style={{color: "var(--accent)"}}>{precision}</span>
                </p>
                <p>
                    &nbsp; Large Detections:{" "}
                    <span style={{color: "var(--accent)"}}>{precision_large}</span>
                </p>
                <p>
                    &nbsp; Small Detections:{" "}
                    <span style={{color: "var(--accent)"}}>{precision_small}</span>
                </p>
                <p>
                    Recall:{" "}
                    <span style={{color: "var(--accent)"}}>{recall}</span>
                </p>
            </div>

            {/* MODEL-WIDE METRICS */}
            {modelMetrics && (
                <div
                    style={{
                        marginTop: 20,
                        backgroundColor: "var(--panel-bg)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        padding: 16,
                        boxShadow: "0 0 12px rgba(0,0,0,0.4)"
                    }}
                >
                    <h3
                        style={{
                            fontSize: 12,
                            textTransform: "uppercase",
                            letterSpacing: 2,
                            marginBottom: 12,
                            color: "var(--text-muted)"
                        }}
                    >Model Metrics:
                    </h3>
                    <p>Precision:
                    </p>
                    <p>&nbsp; Overall:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.model_precision.toFixed(2)}
                        </span>
                    </p>
                    <p>&nbsp; Large Detections:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.precision_large.toFixed(2)}
                        </span>
                    </p>
                    <p>&nbsp; Small Detections:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.precision_small.toFixed(2)}
                        </span>
                    </p>
                    <p>Recall:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.model_recall.toFixed(2)}
                        </span>
                    </p>
                    <p>Mean Average Precision:
                    </p>
                    <p>&nbsp; Overall:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.mAP.toFixed(2)}
                        </span>
                    </p>
                    <p>&nbsp; mAP_50:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.mAP_50.toFixed(2)}
                        </span>
                    </p>
                    <p>&nbsp; mAP_75:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.mAP_75.toFixed(2)}
                        </span>
                    </p>
                    <p>Given confidence threshold:{" "}
                        <span style={{color: "var(--accent)"}}>
                            {modelMetrics.human_review_confidence.toFixed(2)}
                        </span>,{" "}
                        <span style={{color: "var(--warning)"}}>
                            {(modelMetrics.human_review_percentage * 100).toFixed(2)}%
                        </span>
                        {" "}of detections will require a manual review
                    </p>
                </div>
            )}
        </div>
    );
}

export default Report