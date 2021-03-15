import React from "react";

function Output({ response }) {
  const {
    filename,
    output: { word },
    output: { confidences },
  } = response;
  return (
    <div>
      <p>filename={filename}</p>
      <p>Output = {word}</p>
    </div>
  );
}

export default Output;
