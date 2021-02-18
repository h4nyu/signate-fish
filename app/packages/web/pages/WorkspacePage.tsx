import React, { useState } from "react";
import { observer } from "mobx-react-lite";
import { Map } from "immutable";
import FileUpload from "@charpoints/web/components/FileUpload";
import store from "@sivic/web/store";
import { useParams } from "react-router-dom";
import ImageTable from "@sivic/web/components/ImageTable";

const Content = observer(() => {
  const { id } = useParams<{ id: string }>();
  const { workspaceForm, imageProcess } = store;
  const { save } = store.workspaceForm;
  const { workspaces } = store.data.state;
  const { init } = store.data;
  return (
    <div
      className="box"
      style={{
        display: "grid",
        gridTemplateRows: "auto 1fr 110px",
        width: "100%",
        height: "100%",
      }}
    >
      <div
        className="field"
        style={{
          gridRow: "1",
          height: "100%",
        }}
      >
        <label className="label">Name</label>
        <div className="control">
          <input
            className="input"
            type="text"
            value={store.workspaceForm.state.name}
            onChange={(e) => store.workspaceForm.setName(e.target.value)}
          />
        </div>
      </div>
      <div
        style={{
          gridRow: "2",
        }}
      >
        <label className="label">Image List</label>
        <ImageTable
          images={workspaceForm.imageForm.state.images}
          onClick={(id) => imageProcess.init(workspaceForm.state.id, id)}
          onDelete={workspaceForm.imageForm.deleteImage}
        />
      </div>
      <div
        style={{
          gridRow: "3",
        }}
      >
        <FileUpload
          accept={"application/json, image/*"}
          onChange={store.imageForm.uploadFiles}
        />
      </div>
    </div>
  );
});

export default Content;
