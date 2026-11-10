import { useMemo } from "react";
import { buildInsights, type Insight, type InsightVisual } from "../model/insights";
import { matchStats } from "../model/stats";
import { formatClock, formatPercent, formatSplit } from "../model/time";
import type { Run } from "../model/types";
import { C, alpha } from "../render/palette";
import { paintMiniStrip } from "../render/timeline";
import { useApp } from "../state/app";
import { useStaticCanvas } from "../state/hooks";
import { ArrowRightIcon } from "./icons";
import { SourceBar } from "./PossessionTab";

/** A stretch of the timeline in one line, with the part the finding is about marked out. */
function Strip({ run, visual }: { run: Run; visual: Extract<InsightVisual, { kind: "strip" }> }) {
  const ref = useStaticCanvas(
    (ctx, size) => {
      const { width: w, height: h, dpr } = size;
      const barY = 12;
      const barH = 16;
      paintMiniStrip(ctx, 0, barY, w, barH, dpr, run, visual.from, visual.to);
      const span = Math.max(1, visual.to - visual.from);
      const x0 = ((visual.markFrom - visual.from) / span) * w;
      const x1 = ((visual.markTo - visual.from) / span) * w;
      // dim what is outside the marked part, and bracket it
      ctx.fillStyle = "rgba(15, 17, 19, 0.6)";
      ctx.fillRect(0, barY, Math.max(0, x0), barH);
      ctx.fillRect(x1, barY, Math.max(0, w - x1), barH);
      ctx.strokeStyle = alpha(C.ink, 0.9);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const left = Math.max(0.75, x0);
      const right = Math.min(w - 0.75, x1);
      ctx.moveTo(left + 4, barY - 4.5);
      ctx.lineTo(left, barY - 4.5);
      ctx.lineTo(left, barY + barH + 4.5);
      ctx.lineTo(left + 4, barY + barH + 4.5);
      ctx.moveTo(right - 4, barY - 4.5);
      ctx.lineTo(right, barY - 4.5);
      ctx.lineTo(right, barY + barH + 4.5);
      ctx.lineTo(right - 4, barY + barH + 4.5);
      ctx.stroke();
      void h;
    },
    [run, visual.from, visual.to, visual.markFrom, visual.markTo],
  );
  return (
    <div className="viz viz--strip">
      <canvas ref={ref} className="viz__strip" role="img" aria-label="That stretch of the timeline" />
      <div className="viz__ends num">
        <span>{formatClock(visual.from / run.fps)}</span>
        <span>{formatClock(visual.to / run.fps)}</span>
      </div>
    </div>
  );
}

function Visual({ run, insight }: { run: Run; insight: Insight }) {
  const visual = insight.visual;
  if (visual.kind === "split") {
    return (
      <div className="viz">
        {visual.rows.map((row) => {
          const [h, a] = formatSplit(row.home, row.away);
          const total = row.home + row.away;
          return (
            <div className="split" key={row.label}>
              <span className="split__label">{row.label}</span>
              <span className="split__value split__value--home num">{h}</span>
              <span className="split__bar" role="img" aria-label={`${run.home} ${h}, ${run.away} ${a}`}>
                <span className="split__home" style={{ width: `${total ? (row.home / total) * 100 : 0}%` }} />
                <span className="split__away" style={{ width: `${total ? (row.away / total) * 100 : 0}%` }} />
              </span>
              <span className="split__value split__value--away num">{a}</span>
            </div>
          );
        })}
      </div>
    );
  }
  if (visual.kind === "thirds") {
    const max = Math.max(1, ...visual.counts);
    const words = ["Own third", "Middle third", "Final third"];
    const tone = visual.team === 0 ? "home" : "away";
    return (
      <div className={`viz viz--thirds viz--${tone}`} role="img" aria-label={`${visual.counts[0]} in their own third, ${visual.counts[1]} in the middle third, ${visual.counts[2]} in the final third`}>
        <div className="thirds">
          {visual.counts.map((count, i) => (
            <div key={words[i]} className={i === visual.pick ? "thirds__zone thirds__zone--pick" : "thirds__zone"}>
              <span className="thirds__fill" style={{ opacity: 0.1 + 0.55 * (count / max) }} />
              <span className="thirds__count num">{count}</span>
              <span className="thirds__name">{words[i]}</span>
            </div>
          ))}
        </div>
        <p className="thirds__dir">
          {visual.team === 0 ? run.home : run.away} attacking <ArrowRightIcon size={13} />
        </p>
      </div>
    );
  }
  if (visual.kind === "bands") {
    return (
      <div className="viz viz--bars" role="img" aria-label="Share of each quarter-hour with the ball in play">
        {visual.values.map((value, i) => (
          <div key={visual.labels[i]} className={i === visual.pick ? "bar bar--pick" : "bar"}>
            <span className="bar__value num">{value === value ? formatPercent(value) : "–"}</span>
            <span className="bar__col">
              <span className="bar__fill" style={{ height: `${value === value ? value * 100 : 0}%` }} />
            </span>
            <span className="bar__label num">{visual.labels[i]}</span>
          </div>
        ))}
      </div>
    );
  }
  return <Strip run={run} visual={visual} />;
}

/** Findings written from the run, each with a way to go and see it. */
export function InsightsTab() {
  const { run, goTo } = useApp();
  const insights = useMemo(() => buildInsights(run, matchStats(run)), [run]);

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <p className="label">Insights</p>
          <h2 className="page__title">What stands out</h2>
        </div>
        <p className="page__meta">Worked out from this run. Each one takes you to the moment on the timeline.</p>
      </header>
      {insights.length === 0 ? (
        <section className="panel empty">
          <p className="empty__title">Nothing to report yet</p>
          <p className="empty__text">There are no possession answers for this match. Open its run file and the findings appear here.</p>
        </section>
      ) : (
        <div className="ins">
          {insights.map((insight, i) => (
            <article className="panel card" key={insight.id}>
              <p className="card__kicker">
                <span className="card__num num">{String(i + 1).padStart(2, "0")}</span>
                {insight.kicker}
              </p>
              <h3 className="card__headline">{insight.headline}</h3>
              <p className="card__detail">{insight.detail}</p>
              <Visual run={run} insight={insight} />
              <button type="button" className="card__link" onClick={() => goTo(insight.frame, "possession")}>
                <span>{insight.linkLabel}</span>
                <span className="card__at num">{formatClock(insight.frame / run.fps)}</span>
                <ArrowRightIcon size={16} />
              </button>
            </article>
          ))}
        </div>
      )}
      <SourceBar />
    </div>
  );
}
