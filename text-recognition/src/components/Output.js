import React, { useEffect } from "react";

function Output({ response }) {
  useEffect(() => {
    console.log(response);
  }, [response]);

  const { filename, output } = response;

  return (
    <div>
      <p>filename={filename}</p>
      <p className="display-linebreak">Output={output}</p>
    </div>
  );
}

export default Output;
