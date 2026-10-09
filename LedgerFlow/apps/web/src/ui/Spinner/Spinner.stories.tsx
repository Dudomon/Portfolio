import type { Meta, StoryObj } from '@storybook/react';

import { Spinner } from './Spinner';

const meta = {
  title: 'UI/Spinner',
  component: Spinner,
} satisfies Meta<typeof Spinner>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Decorative: Story = {};
export const Announced: Story = { args: { label: 'Loading documents' } };
export const Small: Story = { args: { size: 'sm' } };
