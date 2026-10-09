import type { SVGProps } from 'react';

type IconProps = Omit<SVGProps<SVGSVGElement>, 'children'>;

/** 24×24 stroke icons. Decorative by default; label the surrounding control instead. */
function createIcon(displayName: string, paths: readonly string[]) {
  function Icon(props: IconProps) {
    return (
      <svg
        width="1em"
        height="1em"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        focusable="false"
        {...props}
      >
        {paths.map((d) => (
          <path key={d} d={d} />
        ))}
      </svg>
    );
  }
  Icon.displayName = displayName;
  return Icon;
}

export const CloseIcon = createIcon('CloseIcon', ['M18 6 6 18', 'M6 6l12 12']);
export const CheckIcon = createIcon('CheckIcon', ['M20 6 9 17l-5-5']);
export const ChevronUpIcon = createIcon('ChevronUpIcon', ['m18 15-6-6-6 6']);
export const ChevronDownIcon = createIcon('ChevronDownIcon', ['m6 9 6 6 6-6']);
export const ChevronLeftIcon = createIcon('ChevronLeftIcon', ['m15 18-6-6 6-6']);
export const ChevronRightIcon = createIcon('ChevronRightIcon', ['m9 18 6-6-6-6']);
export const SortIcon = createIcon('SortIcon', ['m7 15 5 5 5-5', 'm7 9 5-5 5 5']);
export const UploadIcon = createIcon('UploadIcon', [
  'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4',
  'm17 8-5-5-5 5',
  'M12 3v12',
]);
export const SearchIcon = createIcon('SearchIcon', [
  'm21 21-4.3-4.3',
  'M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z',
]);
export const RefreshIcon = createIcon('RefreshIcon', [
  'M3 12a9 9 0 0 1 15-6.7L21 8',
  'M21 3v5h-5',
  'M21 12a9 9 0 0 1-15 6.7L3 16',
  'M8 16H3v5',
]);
export const FileIcon = createIcon('FileIcon', [
  'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z',
  'M14 2v6h6',
]);
export const MoonIcon = createIcon('MoonIcon', ['M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z']);
export const SunIcon = createIcon('SunIcon', [
  'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z',
  'M12 2v2',
  'M12 20v2',
  'm4.9 4.9 1.4 1.4',
  'm17.7 17.7 1.4 1.4',
  'M2 12h2',
  'M20 12h2',
  'm6.3 17.7-1.4 1.4',
  'm19.1 4.9-1.4 1.4',
]);
export const PlusIcon = createIcon('PlusIcon', ['M12 5v14', 'M5 12h14']);
export const TrashIcon = createIcon('TrashIcon', [
  'M3 6h18',
  'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6',
  'M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2',
]);
