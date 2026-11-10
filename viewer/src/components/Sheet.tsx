import { useId, useLayoutEffect, useRef, type ReactNode } from "react";
import { CrossIcon } from "./icons";

/**
 * A panel over the page: a centred dialog on a desktop, a bottom sheet on a phone.
 * A native <dialog> opened with showModal(), so the page behind is inert and focus stays inside.
 */
export function Sheet({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  wide = false,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descId = useId();

  useLayoutEffect(() => {
    const dialog = ref.current;
    if (!dialog || !open) return;
    // remember what opened the sheet, so focus can go back there when it closes
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!dialog.open) {
      try {
        dialog.showModal();
      } catch {
        dialog.setAttribute("open", "");
      }
    }
    return () => {
      if (dialog.open) dialog.close();
      if (opener && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, [open]);

  if (!open) return null;
  return (
    <dialog
      ref={ref}
      className={wide ? "sheet sheet--wide" : "sheet"}
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      onClose={onClose}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        // a click on the dimmed backdrop (the dialog element itself, outside its panel) closes it
        if (event.target === ref.current) onClose();
      }}
    >
      <div className="sheet__panel">
        <header className="sheet__head">
          <div>
            <h2 className="sheet__title" id={titleId}>
              {title}
            </h2>
            {description ? (
              <p className="sheet__desc" id={descId}>
                {description}
              </p>
            ) : null}
          </div>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <CrossIcon size={18} />
          </button>
        </header>
        <div className="sheet__body">{children}</div>
        {footer ? <footer className="sheet__foot">{footer}</footer> : null}
      </div>
    </dialog>
  );
}
