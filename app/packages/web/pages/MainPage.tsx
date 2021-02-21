import React, { useState } from "react";
import { observer } from "mobx-react-lite";
import { Map } from "immutable";
import WorkspaceTable from "@sivic/web/components/WorkspaceTable";
import store from "@sivic/web/store";

const Content = observer(() => {
  const { workspaces } = store.data.state;
  const { init } = store.data;
  return (
    <div
      className="box"
      style={{
        height: "100%",
        overflow: "scroll",
      }}
    >
      <WorkspaceTable
        workspaces={workspaces.toList().toJS()}
        onClick={store.workspaceForm.update}
        onDelete={store.workspaceForm.delete}
      />
    </div>
  );
});

export default Content;
