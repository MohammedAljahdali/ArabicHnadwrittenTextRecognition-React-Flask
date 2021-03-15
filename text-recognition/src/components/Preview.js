import React from "react";

function Preview({ image }) {
  return (
    <div>
      {image ? (
        <img src={URL.createObjectURL(image)} alt={image.name} />
      ) : (
        <p>Upload Image</p>
      )}
    </div>
  );
}

export default Preview;
