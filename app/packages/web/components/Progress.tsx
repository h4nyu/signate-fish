import React from "react";

export default function Progress(props: { isActive?: boolean }) {
  return (
    <>
      {props.isActive && (
        <progress
          className="progress is-small is-primary"
          max="100"
          style={{
            margin: 0,
            borderRadius: 0,
            height: 10,
            position: "fixed",
            top: 0,
            left: 0,
            zIndex: 100,
          }}
        />
      )}
    </>
  );
}
