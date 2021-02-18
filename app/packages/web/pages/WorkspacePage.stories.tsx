import React from "react";
import WorkspacePage from "./WorkspacePage";
import { Router, Switch, Route } from "react-router-dom";
import { createHashHistory } from "history";
import { Workspace } from "@sivic/core/workspace";
import store from "@sivic/web/store";

const history = createHashHistory();
export default {
  title: "WorkspacePage",
  component: WorkspacePage,
  decorators: [
    (Story) => (
      <Router history={history}>
        <div style={{ height: 500 }}>
          <Story />
        </div>
      </Router>
    ),
  ],
};

const form = store.workspaceForm;
form.setName("testname");

export const Primary = (args) => <WorkspacePage />;
