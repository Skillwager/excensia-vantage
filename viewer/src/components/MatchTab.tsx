import { useMemo } from "react";
import { countLabels } from "../model/derive";
import { matchStats, type SpellRef } from "../model/stats";
import { formatClock, formatDuration, formatPercent, formatSplit, formatTotalClock } from "../model/time";
import { A, D, H, X } from "../model/types";
import { useApp } from "../state/app";
import { ArrowRightIcon } from "./icons";
import { SourceBar } from "./PossessionTab";

function SplitBar({ home, away, label }: { home: number; away: number; label?: string }) {
  const total = home + away;
  const [h, a] = formatSplit(home, away);
  const share = total > 0 ? (home / total) * 100 : 50;
  return (
    <div className="split">
      {label ? <span className="split__label">{label}</span> : null}
      <span className="split__value split__value--home num">{h}</span>
      <span className="split__bar" role="img" aria-label={`Home ${h}, Away ${a}`}>
        <span className="split__home" style={{ width: `${total > 0 ? share : 0}%` }} />
        <span className="split__away" style={{ width: `${total > 0 ? 100 - share : 0}%` }} />
      </span>
      <span className="split__value split__value--away num">{a}</span>
    </div>
  );
}

/** The match in numbers: who had the ball, how long it was in play, and when. */
export function MatchTab() {
  const { run, goTo } = useApp();
  const stats = useMemo(() => matchStats(run), [run]);
  const fps = run.fps;
  const live = stats.liveFrames;
  const [homePct, awayPct] = formatSplit(stats.frames[H], stats.frames[A]);
  const watch = (spell: SpellRef) => goTo(Math.max(0, spell.start - Math.round(2 * fps)), "possession");
  // Home's share of the ball in each three minutes of the match
  const flow = useMemo(() => {
    const step = Math.max(1, Math.round(3 * 60 * fps));
    const out: { start: number; share: number }[] = [];
    for (let start = 0; start < run.frames; start += step) {
      const counts = countLabels(run, start, Math.min(run.frames, start + step));
      const inPlay = counts[H] + counts[A];
      out.push({ start, share: inPlay >= fps * 8 ? counts[H] / inPlay : NaN });
    }
    return out;
  }, [run, fps]);
  const minutes = run.frames / fps / 60;
  const marks: number[] = [];
  for (let m = 0; m <= minutes; m += minutes > 50 ? 15 : minutes > 20 ? 5 : 1) marks.push(m);
  const empty = live === 0 && stats.frames[D] === 0;

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <p className="label">Match summary</p>
          <h2 className="page__title">{run.title}</h2>
        </div>
        <p className="page__meta">
          <span className="num">{formatTotalClock(run.frames, fps)}</span> of footage
          {run.periods.length ? ` · ${run.periods.map((p) => p.name).join(", ")}` : ""}
        </p>
      </header>

      {empty ? (
        <section className="panel empty">
          <p className="empty__title">No answers for this match yet</p>
          <p className="empty__text">Open the run file for this match and its summary appears here.</p>
        </section>
      ) : (
        <div className="mt">
          <section className="panel mt__share" aria-label="Share of the ball">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">Share of the ball</h3>
              <p className="panel-note">of the {formatDuration(live / fps)} the ball was in play</p>
            </header>
            <div className="share">
              <div className="share__side">
                <span className="share__figure share__figure--home">{homePct}</span>
                <span className="share__name">{run.home}</span>
              </div>
              <div className="share__side share__side--away">
                <span className="share__figure share__figure--away">{awayPct}</span>
                <span className="share__name">{run.away}</span>
              </div>
              <div className="share__bar" role="img" aria-label={`${run.home} ${homePct}, ${run.away} ${awayPct}`}>
                <span className="split__home" style={{ width: `${live > 0 ? (stats.frames[H] / live) * 100 : 0}%` }} />
                <span className="split__away" style={{ width: `${live > 0 ? (stats.frames[A] / live) * 100 : 0}%` }} />
              </div>
            </div>
            <div className="flow">
              <p className="flow__title">
                <span className="label">Through the match</span>
                <span className="flow__note">three minutes at a time</span>
              </p>
              <div className="flow__plot" role="group" aria-label="Share of the ball through the match, three minutes at a time">
                {flow.map((bucket) => (
                  <button
                    key={bucket.start}
                    type="button"
                    className="flow__col"
                    title={
                      bucket.share === bucket.share
                        ? `${formatClock(bucket.start / fps)}: ${run.home} ${formatPercent(bucket.share)}, ${run.away} ${formatPercent(1 - bucket.share)}`
                        : `${formatClock(bucket.start / fps)}: the ball was hardly in play`
                    }
                    aria-label={`Go to ${formatClock(bucket.start / fps)}`}
                    tabIndex={-1}
                    onClick={() => goTo(bucket.start, "possession")}
                  >
                    {bucket.share === bucket.share ? (
                      <>
                        <span className="flow__away" style={{ flexGrow: 1 - bucket.share }} />
                        <span className="flow__home" style={{ flexGrow: bucket.share }} />
                      </>
                    ) : (
                      <span className="flow__none" />
                    )}
                  </button>
                ))}
                <span className="flow__half" aria-hidden="true" />
              </div>
              <div className="flow__axis num" aria-hidden="true">
                {marks.map((m) => (
                  <span key={m} style={{ left: `${(m / minutes) * 100}%` }}>
                    {`${m}'`}
                  </span>
                ))}
              </div>
            </div>
            {stats.periods.length > 1 ? (
              <div className="share__periods">
                {stats.periods.map((p) => (
                  <SplitBar key={p.name} label={p.name} home={p.counts[H]} away={p.counts[A]} />
                ))}
              </div>
            ) : null}
          </section>

          <section className="panel mt__play" aria-label="Ball in play">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">Ball in play</h3>
            </header>
            <div className="inplay">
              <p className="inplay__figure">
                <span className="figure">{formatPercent(stats.inPlayShare)}</span>
                <span className="inplay__caption">of the match with the ball in play</span>
              </p>
              <div className="inplay__bar" role="img" aria-label={`In play ${formatPercent(stats.inPlayShare)}`}>
                <span className="inplay__live" style={{ width: `${(live / run.frames) * 100}%` }} />
                <span className="inplay__dead" style={{ width: `${(stats.frames[D] / run.frames) * 100}%` }} />
                <span className="inplay__none" style={{ width: `${(stats.frames[X] / run.frames) * 100}%` }} />
              </div>
              <dl className="facts">
                <div>
                  <dt>
                    <span className="legend__dot legend__dot--live" />
                    In play
                  </dt>
                  <dd>{formatDuration(live / fps)}</dd>
                </div>
                <div>
                  <dt>
                    <span className="legend__dot legend__dot--dead" />
                    Dead ball
                  </dt>
                  <dd>{formatDuration(stats.frames[D] / fps)}</dd>
                </div>
                {stats.frames[X] > 0 ? (
                  <div>
                    <dt>
                      <span className="legend__hatch" />
                      No answer
                    </dt>
                    <dd>{formatDuration(stats.frames[X] / fps)}</dd>
                  </div>
                ) : null}
                <div>
                  <dt>Stoppages</dt>
                  <dd>
                    {stats.dead.spells}
                    {stats.dead.spells ? <span className="facts__aside"> · {formatDuration(stats.dead.meanSeconds)} each on average</span> : null}
                  </dd>
                </div>
              </dl>
              {stats.dead.longest ? (
                <button type="button" className="rowlink" onClick={() => watch(stats.dead.longest!)}>
                  <span>
                    Longest stoppage: <strong>{formatDuration(stats.dead.longest.seconds)}</strong> from <span className="num">{formatClock(stats.dead.longest.start / fps)}</span>
                  </span>
                  <ArrowRightIcon size={16} />
                </button>
              ) : null}
            </div>
          </section>

          <section className="panel mt__bands" aria-label="By 15 minutes">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">By 15 minutes</h3>
              <ul className="legend">
                <li>
                  <span className="legend__dot legend__dot--home" />
                  {run.home}
                </li>
                <li>
                  <span className="legend__dot legend__dot--away" />
                  {run.away}
                </li>
                <li>
                  <span className="legend__dot legend__dot--dead" />
                  Dead ball
                </li>
              </ul>
            </header>
            <div className="bands">
              {stats.bands.map((band) => {
                const total = band.end - band.start;
                const [h, a] = formatSplit(band.counts[H], band.counts[A]);
                const part = (n: number) => `${(n / total) * 100}%`;
                return (
                  <button
                    key={band.start}
                    type="button"
                    className="band"
                    onClick={() => goTo(band.start, "possession")}
                    title={`${run.home} ${formatPercent(band.counts[H] / total)}, ${run.away} ${formatPercent(band.counts[A] / total)}, dead ball ${formatPercent(band.counts[D] / total)}. Go to ${Math.round(band.fromMinute)}'.`}
                  >
                    <span className="band__split num">
                      <span className="band__h">{h}</span>
                      <span className="band__a">{a}</span>
                    </span>
                    <span className="band__col">
                      <span className="band__none" style={{ height: part(band.counts[X]) }} />
                      <span className="band__dead" style={{ height: part(band.counts[D]) }} />
                      <span className="band__away" style={{ height: part(band.counts[A]) }} />
                      <span className="band__home" style={{ height: part(band.counts[H]) }} />
                    </span>
                    <span className="band__label num">
                      {Math.round(band.fromMinute)}'–{Math.round(band.toMinute)}'
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="panel mt__spells" aria-label="Possessions">
            <header className="panel-head panel-head--plain">
              <h3 className="panel-title">Possessions</h3>
              <p className="panel-note">{stats.turnovers} turnovers in open play</p>
            </header>
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Team</th>
                  <th scope="col">Number</th>
                  <th scope="col">Average</th>
                  <th scope="col">Longest</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    [run.home, stats.home, "home"],
                    [run.away, stats.away, "away"],
                  ] as const
                ).map(([name, team, tone]) => (
                  <tr key={tone}>
                    <th scope="row">
                      <span className={`legend__dot legend__dot--${tone}`} />
                      {name}
                    </th>
                    <td>{team.spells}</td>
                    <td>{team.spells ? formatDuration(team.meanSeconds) : "–"}</td>
                    <td>{team.longest ? formatDuration(team.longest.seconds) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4 className="label mt__sub">Longest spells on the ball</h4>
            <ol className="spells">
              {stats.longestSpells.slice(0, 3).map((spell) => (
                <li key={spell.start}>
                  <button type="button" className="rowlink" onClick={() => watch(spell)}>
                    <span className="spells__who">
                      <span className={`legend__dot legend__dot--${spell.label === H ? "home" : "away"}`} />
                      {spell.label === H ? run.home : run.away}
                    </span>
                    <span className="spells__len">{formatDuration(spell.seconds)}</span>
                    <span className="spells__at">
                      from <span className="num">{formatClock(spell.start / fps)}</span>
                    </span>
                    <ArrowRightIcon size={16} />
                  </button>
                </li>
              ))}
            </ol>
          </section>
        </div>
      )}
      <SourceBar />
    </div>
  );
}
