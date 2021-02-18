import React from "react";
import PageLayout from "./PageLayout";
import Mock from "@sivic/web/components/Mock";
import { Workspace } from "@sivic/core/workspace";

export default {
  title: "PageLayout",
  component: PageLayout,
};

export const Primary = (args) => (
  <PageLayout
    header={<Mock name="header" />}
    content={<Mock name="content" style={{ height: 2000 }} />}
    sidebar={<Mock name="sidebar" style={{ height: 2000 }} />}
  />
);
