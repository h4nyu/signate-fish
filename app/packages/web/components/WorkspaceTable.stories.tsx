import React from "react";
import WorkspaceTable from "./WorkspaceTable";
import { Workspace } from "@sivic/core/workspace";

export default {
  title: "WorkspaceTable",
  component: WorkspaceTable,
};

const workspaces = [{ ...Workspace() }];
export const Primary = (args) => (
  <WorkspaceTable {...args} workspaces={workspaces} />
);
