import React from "react";
import { formatDistance } from "date-fns";
export const DateView = (props: { value: Date }) => {
  const now = new Date();
  return <>{formatDistance(now, props.value, { addSuffix: true })}</>;
};
export default DateView;
