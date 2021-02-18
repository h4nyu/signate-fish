import React from "react";
import Sidebar from "./Sidebar";
import { Router, Switch, Route } from "react-router-dom";
import { createHashHistory } from "history";

const history = createHashHistory();
export default {
  title: "Sidebar",
  component: Sidebar,
  decorators: [
    (Story) => (
      <Router history={history}>
        <Story />
      </Router>
    ),
  ],
};

export const Primary = (args) => <Sidebar />;
