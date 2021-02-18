import React, { useState, useEffect } from "react";
import { observer } from "mobx-react-lite";
import { Map } from "immutable";
import ImageView from "@sivic/web/components/ImageView";
import store from "@sivic/web/store";

const Content = observer(() => {
  const { imageProcess } = store;
  const { image } = imageProcess.state;
  const { init } = store.data;
  return (
    <div
      className="box"
      style={{
        display: "grid",
        gridTemplateRows: "1fr 110px",
        width: "100%",
        height: "100%",
      }}
    >
      {image && image.data && <ImageView imageData={image.data} size={512} />}
    </div>
  );
});

export default Content;
