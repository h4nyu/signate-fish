import React, { useRef, useLayoutEffect, useState } from "react";
export const PageLayout = (props: {
  header?: React.ReactNode;
  content?: React.ReactNode;
  sidebar?: React.ReactNode;
}) => {
  const { header, content, sidebar } = props;

  return (
    <div
      style={{
        height: "100vh",
        display: "grid",
        gridTemplateRows: "auto 1fr",
        gridTemplateColumns: "auto 1fr",
      }}
    >
      <div
        style={{
          gridRow: "1",
          gridColumn: "1 / span 2",
        }}
      >
        {header}
      </div>
      <div
        style={{
          gridRow: "2",
          gridColumn: "2",
          overflow: "auto",
        }}
      >
        {content}
      </div>
      <div
        style={{
          gridRow: "2",
          gridColumn: "1",
          overflow: "auto",
        }}
      >
        {sidebar}
      </div>
    </div>
  );
};
export default PageLayout;
