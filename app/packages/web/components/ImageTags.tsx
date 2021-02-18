import React from "react";
import { Image, ImageTag } from "@sivic/core/image";

const Tags = (props: { image: Image; onClick?: (id: string) => void }) => {
  const { image, onClick } = props;
  const tag = image.tag;
  return (
    <div className="buttons has-addons">
      <button
        className={tag === "Source" ? "button is-success" : "button"}
        onClick={() => onClick && onClick(image.id)}
      >
        Source
      </button>
      <button
        className={tag === "Target" ? "button is-info" : "button"}
        onClick={() => onClick && onClick(image.id)}
      >
        Target
      </button>
    </div>
  );
};
export default Tags;
