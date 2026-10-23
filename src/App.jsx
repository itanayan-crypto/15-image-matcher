import { useEffect } from "react";
import ImageMatcher from "./components/ImageMatcher/ImageMatcher.jsx";

// body[data-task] is used by styles/base.css for this task's page background.
const TASK_ID = "15";

export default function App() {
  useEffect(() => {
    document.body.dataset.task = TASK_ID;
    return () => { delete document.body.dataset.task; };
  }, []);

  return (
    <main className="page">
      <div className="page-head">
        <span className="task-chip">Task 15</span>
        <h1 className="page-title">Image Matching</h1>
        <p className="page-sub">Upload your latest image and it is matched against the image DB folder with an image matching algorithm, followed by an analytical report.</p>
      </div>
      <ImageMatcher />
    </main>
  );
}
