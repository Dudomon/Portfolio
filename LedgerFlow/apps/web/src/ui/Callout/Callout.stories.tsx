import type { Meta, StoryObj } from '@storybook/react';

import { Button } from '../Button/Button';
import { Callout } from './Callout';

const meta = {
  title: 'UI/Callout',
  component: Callout,
  args: {
    title: 'Extraction is still running',
    children: 'Fields will appear here once the worker finishes.',
  },
} satisfies Meta<typeof Callout>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Info: Story = {};
export const Success: Story = { args: { tone: 'success', title: 'Approved' } };
export const Warning: Story = {
  args: {
    tone: 'warning',
    title: 'This document changed since you opened it',
    actions: <Button size="sm">Reload</Button>,
  },
};
export const Danger: Story = {
  args: {
    tone: 'danger',
    title: 'Documents could not be loaded',
    children: undefined,
    actions: <Button size="sm">Retry</Button>,
  },
};
