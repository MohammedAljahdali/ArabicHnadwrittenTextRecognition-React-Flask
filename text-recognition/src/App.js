import React, { useEffect } from "react";
import Preview from "./components/Preview";
import Output from "./components/Output";
import logo from "./logo.svg";
import { useState } from "react";
import "./App.css";
const axios = require("axios");
const api = axios.create({ baseURL: "http://127.0.0.1:5000" });

function App() {
  const [image, setImage] = useState(null);
  const [singleImage, setSingleImage] = useState(true);
  const [apiResponse, setApiResponse] = useState(null);

  useEffect(() => {
    callAPI();
  }, []);

  const callAPI = () => {
    api
      .get("/")
      .then(function (response) {
        // handle success
        console.log(response);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
      .then(function () {
        // always executed
      });
  };

  const predicte = () => {
    console.log(image);
    const formData = new FormData();
    formData.append("image", image);
    formData.append("singleImage", singleImage);
    api
      .post("/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then(function (response) {
        console.log(response.data);
        console.log(response.data.output);
        const { filename, output } = response.data;
        console.log(filename, output);
        setApiResponse(response.data);
      })
      .catch(function (error) {
        console.log(error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <Preview image={image} />
        {apiResponse && <Output response={apiResponse} />}
        <input
          id={"image"}
          type={"file"}
          accept={"image/*"}
          onChange={(e) => {
            setImage(e.target.files[0]);
            console.log("hello?");
            console.log("can you hear me?");
          }}
        />
        <p>
          Image contain multiple words?{" "}
          <input
            type="checkbox"
            checked={!singleImage}
            value={!singleImage}
            onChange={(e) => {
              setSingleImage(!e.currentTarget.checked);
              console.log(singleImage);
            }}
          />
        </p>
        <button onClick={predicte}>Predicte</button>
      </header>
    </div>
  );
}

export default App;
