import { useRef, type KeyboardEvent } from "react";
import wordmarkUrl from "../assets/wordmark.png";
import type { TabId } from "../state/app";
import { SettingsIcon } from "./icons";

export const TABS: { id: TabId; label: string }[] = [
  { id: "match", label: "Match" },
  { id: "possession", label: "Possession" },
  { id: "shape", label: "Team shape" },
  { id: "insights", label: "Insights" },
];

export function Header({ tab, onTab, onSettings }: { tab: TabId; onTab: (tab: TabId) => void; onSettings: () => void }) {
  const listRef = useRef<HTMLDivElement>(null);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const at = TABS.findIndex((t) => t.id === tab);
    let next = at;
    if (event.key === "ArrowRight") next = (at + 1) % TABS.length;
    else if (event.key === "ArrowLeft") next = (at + TABS.length - 1) % TABS.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TABS.length - 1;
    else return;
    event.preventDefault();
    event.stopPropagation();
    onTab(TABS[next].id);
    const button = listRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next];
    button?.focus();
    button?.scrollIntoView({ block: "nearest", inline: "nearest" });
  };

  return (
    <header className="hdr">
      <div className="hdr__brand">
        <img className="hdr__wordmark" src={wordmarkUrl} alt="Excensia" width={544} height={65} draggable={false} />
        <span className="hdr__rule" aria-hidden="true" />
        <div className="hdr__product">
          <span className="hdr__num">01</span>
          <h1 className="hdr__title">Possession Lab</h1>
        </div>
        <span className="hdr__rule hdr__rule--tagline" aria-hidden="true" />
        <p className="hdr__tagline">See who controls the game</p>
      </div>
      <div className="hdr__nav">
        <div className="tabs" role="tablist" aria-label="Views" ref={listRef} onKeyDown={onKeyDown}>
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls="tabpanel"
              tabIndex={tab === t.id ? 0 : -1}
              className="tabs__tab"
              onClick={() => onTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <span className="hdr__rule hdr__rule--nav" aria-hidden="true" />
        <button type="button" className="icon-btn hdr__settings" aria-label="Files and settings" title="Files and settings" onClick={onSettings}>
          <SettingsIcon size={19} />
        </button>
      </div>
    </header>
  );
}
