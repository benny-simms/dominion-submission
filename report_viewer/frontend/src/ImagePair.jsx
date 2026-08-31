import React, {useState, useEffect} from "react";
import "../index.css";

function ImagePair({ id, modelVersion}) {
  const [gtSrc, setGtSrc] = useState("");
  const [predSrc, setPredSrc] = useState("");

  useEffect(() => {
    // Build URLs fresh each time dataset changes
    const gtUrl = `http://localhost:8000/ground_truth/${id}`;
    const predUrl = `http://localhost:8000/predictions/${id}`;

    setGtSrc(gtUrl);
    setPredSrc(predUrl);
  }, [id, modelVersion]);

  return (
    <div
      style={{
        backgroundColor: "var(--panel-bg)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center"
      }}
    >
      {/* Images side-by-side */}
      <div
        style={{
          display: "flex",
          gap: "6px",
          padding: "6px",
          backgroundColor: "var(--panel-bg)"
        }}
      >
        {/* Ground Truth block */}
        <div style={{ textAlign: "center" }}>
          <h4
            style={{
              fontSize: 12,
              textTransform: "uppercase",
              letterSpacing: 2,
              marginBottom: 6,
              color: "var(--text-muted)"
            }}
          >
            Ground Truth
          </h4>

          <img
            src={gtSrc}
            alt="ground truth"
            style={{
              maxWidth: "680px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              boxShadow: "0 0 12px rgba(0,0,0,0.6)",
              backgroundColor: "transparent",
              margin: 0,
              padding: 0
            }}
          />
        </div>

        {/* Prediction block */}
        <div style={{ textAlign: "center" }}>
          <h4
            style={{
              fontSize: 12,
              textTransform: "uppercase",
              letterSpacing: 2,
              marginBottom: 6,
              color: "var(--text-muted)"
            }}
          >
            Prediction
          </h4>

          <img
            src={predSrc}
            alt="prediction"
            style={{
              maxWidth: "680px",
              borderRadius: 4,
              border: "1px solid var(--border)",
              boxShadow: "0 0 12px rgba(0,0,0,0.6)",
              backgroundColor: "transparent",
              margin: 0,
              padding: 0
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default ImagePair