import React, { useState, useEffect } from "react";

function Report({ id, modelVersion}) {
  const [report, setReport] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:8000/reports/${id}`)
      .then(res => res.json())
      .then(setReport)
      .catch(err => console.error("Report fetch error:", err));
  }, [id, modelVersion]);

  if (!report) return <div>Loading report…</div>;

  const recall = report.recall.toFixed(2);
  const precision = report.precision.toFixed(2);

  return (
    <div>
      <h3
        style={{
          fontSize: 13,
          textTransform: "uppercase",
          letterSpacing: 2,
          marginBottom: 12
        }}
      >
        Metrics:
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
          <p style={{ color: "var(--text-muted)", fontSize: 11 }}>Total Ground Truth</p>
          <p style={{ fontSize: 14 }}>{report.n_ground_truth}</p>
        </div>
        <div>
          <p style={{ color: "var(--text-muted)", fontSize: 11 }}>Detections</p>
          <p style={{ fontSize: 14 }}>{report.n_detections}</p>
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
          Classification
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
          Performance
        </h4>

        <p>
          Precision:{" "}
          <span style={{ color: "var(--accent)" }}>{precision}</span>
        </p>
        <p>
          Recall:{" "}
          <span style={{ color: "var(--accent)" }}>{recall}</span>
        </p>
      </div>
    </div>
  );
}

export default Report