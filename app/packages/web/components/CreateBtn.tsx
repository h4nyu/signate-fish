import React from "react";

const CreateBtn = (props) => {
  return (
    <button className="button is-success is-light" {...props}>
      <span className="icon is-small">
        <i className="fas fa-plus"></i>
      </span>
    </button>
  );
};
export default CreateBtn;
