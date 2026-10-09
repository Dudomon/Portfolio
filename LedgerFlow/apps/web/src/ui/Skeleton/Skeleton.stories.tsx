import type { Meta, StoryObj } from '@storybook/react';

import { Skeleton, SkeletonText } from './Skeleton';

const meta = {
  title: 'UI/Skeleton',
  component: Skeleton,
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 320 }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof Skeleton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Text: Story = {};
export const Rect: Story = { args: { shape: 'rect', height: 120 } };
export const Circle: Story = { args: { shape: 'circle', width: 40, height: 40 } };
export const Paragraph: Story = { render: () => <SkeletonText lines={4} /> };
