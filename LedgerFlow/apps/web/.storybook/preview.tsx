import type { Preview } from '@storybook/react';

import '../src/styles/global.css';

const preview: Preview = {
  globalTypes: {
    theme: {
      description: 'Color theme',
      toolbar: {
        title: 'Theme',
        icon: 'mirror',
        items: [
          { value: 'light', title: 'Light' },
          { value: 'dark', title: 'Dark' },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: { theme: 'light' },
  decorators: [
    (Story, context) => {
      document.documentElement.dataset.theme = String(context.globals.theme ?? 'light');
      return <Story />;
    },
  ],
  parameters: {
    layout: 'padded',
    controls: { expanded: true },
  },
};

export default preview;
