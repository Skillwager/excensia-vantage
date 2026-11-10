import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 18, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const PlayIcon = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <path d="M7.5 4.6v14.8a.9.9 0 0 0 1.38.76l11.4-7.4a.9.9 0 0 0 0-1.52L8.88 3.84a.9.9 0 0 0-1.38.76Z" />
  </Icon>
);

export const PauseIcon = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <rect x="5.5" y="4" width="4.6" height="16" rx="1.2" />
    <rect x="13.9" y="4" width="4.6" height="16" rx="1.2" />
  </Icon>
);

export const BackIcon = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <path d="M11.6 6.1v11.8a.7.7 0 0 1-1.1.58l-8-5.9a.7.7 0 0 1 0-1.16l8-5.9a.7.7 0 0 1 1.1.58Z" />
    <path d="M22 6.1v11.8a.7.7 0 0 1-1.1.58l-8-5.9a.7.7 0 0 1 0-1.16l8-5.9a.7.7 0 0 1 1.1.58Z" />
  </Icon>
);

export const ForwardIcon = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <path d="M12.4 6.1v11.8a.7.7 0 0 0 1.1.58l8-5.9a.7.7 0 0 0 0-1.16l-8-5.9a.7.7 0 0 0-1.1.58Z" />
    <path d="M2 6.1v11.8a.7.7 0 0 0 1.1.58l8-5.9a.7.7 0 0 0 0-1.16l-8-5.9A.7.7 0 0 0 2 6.1Z" />
  </Icon>
);

/** Previous change: a bar and an arrowhead pointing back to it. */
export const PrevChangeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M6 5v14" />
    <path d="M18.5 6.2v11.6a.6.6 0 0 1-.95.5l-7.8-5.8a.6.6 0 0 1 0-1l7.8-5.8a.6.6 0 0 1 .95.5Z" fill="currentColor" stroke="none" />
  </Icon>
);

export const NextChangeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M18 5v14" />
    <path d="M5.5 6.2v11.6a.6.6 0 0 0 .95.5l7.8-5.8a.6.6 0 0 0 0-1l-7.8-5.8a.6.6 0 0 0-.95.5Z" fill="currentColor" stroke="none" />
  </Icon>
);

export const ChevronDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m6 9 6 6 6-6" />
  </Icon>
);

export const SettingsIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);

export const EyeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 19.88 0 1 1 0 0 1 0 .7 10.75 10.75 0 0 1-19.88 0Z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);

export const EyeOffIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10.73 5.08A10.4 10.4 0 0 1 12 5c4.5 0 8.4 2.7 9.94 6.65a1 1 0 0 1 0 .7 10.8 10.8 0 0 1-1.44 2.49" />
    <path d="M14.08 14.16a3 3 0 0 1-4.24-4.24" />
    <path d="M17.48 17.5A10.75 10.75 0 0 1 2.06 12.35a1 1 0 0 1 0-.7 10.75 10.75 0 0 1 4.45-5.14" />
    <path d="m2 2 20 20" />
  </Icon>
);

export const TeamIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="9" cy="8" r="3.2" />
    <path d="M2.5 20a6.5 6.5 0 0 1 13 0" />
    <path d="M15.6 5.1a3.2 3.2 0 0 1 0 5.8" />
    <path d="M18.3 14.6a6.5 6.5 0 0 1 3.2 5.4" />
  </Icon>
);

export const ClockIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9.2" />
    <path d="M12 6.8V12l3.4 2" />
  </Icon>
);

export const DownloadIcon = (p: IconProps) => (
  <Icon {...p} strokeWidth={2.1}>
    <path d="M12 3.5v11" />
    <path d="m7.2 10.2 4.8 4.8 4.8-4.8" />
    <path d="M4.5 16.5v2.2a1.8 1.8 0 0 0 1.8 1.8h11.4a1.8 1.8 0 0 0 1.8-1.8v-2.2" />
  </Icon>
);

export const CheckIcon = (p: IconProps) => (
  <Icon {...p} strokeWidth={2.2}>
    <path d="m5 12.5 4.5 4.5L19 7.5" />
  </Icon>
);

export const CrossIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Icon>
);

export const AlertIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 8.5v4.5" />
    <path d="M12 16.6h.01" />
    <path d="M10.3 3.9 2.4 17.5a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
  </Icon>
);

export const FolderIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 7.2A2.2 2.2 0 0 1 5.2 5h3.6a2 2 0 0 1 1.6.8l.9 1.2a2 2 0 0 0 1.6.8h5.9A2.2 2.2 0 0 1 21 10v7.8a2.2 2.2 0 0 1-2.2 2.2H5.2A2.2 2.2 0 0 1 3 17.8Z" />
  </Icon>
);

export const FilmIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3" y="4.5" width="18" height="15" rx="2.2" />
    <path d="m10 9.2 5 2.8-5 2.8Z" fill="currentColor" stroke="none" />
  </Icon>
);

export const ZoomInIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 5.5v13M5.5 12h13" />
  </Icon>
);

export const ZoomOutIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5.5 12h13" />
  </Icon>
);

export const ArrowRightIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4.5 12h15" />
    <path d="m13.5 6 6 6-6 6" />
  </Icon>
);

export const ReplayIcon = (p: IconProps) => (
  <Icon {...p} stroke="none" fill="currentColor">
    <path d="M7.5 4.6v14.8a.9.9 0 0 0 1.38.76l11.4-7.4a.9.9 0 0 0 0-1.52L8.88 3.84a.9.9 0 0 0-1.38.76Z" />
  </Icon>
);

/** The Excensia mark: four facets of an X. From the app's own logo component. */
export function Mark({ size = 28, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={(size * 340) / 400} viewBox="-200 -170 400 340" className={className} aria-hidden="true" focusable="false">
      <g transform="translate(-627 -465)" fill="currentColor">
        <path d="M440 324H523L622 425L581 465Z" />
        <path d="M731 324H813L669 465L630 425Z" />
        <path d="M440 606H519L622 505L581 465Z" />
        <path d="M733 606H811L669 465L630 505Z" />
      </g>
    </svg>
  );
}
