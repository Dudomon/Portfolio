import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';

import { CloseIcon, RefreshIcon, TrashIcon } from '../icons';
import { IconButton } from './IconButton';

const meta = {
  title: 'UI/IconButton',
  component: IconButton,
  args: { label: 'Refresh', icon: <RefreshIcon />, onClick: fn() },
} satisfies Meta<typeof IconButton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Ghost: Story = {};
export const Secondary: Story = { args: { variant: 'secondary' } };
export const Small: Story = { args: { size: 'sm', label: 'Close', icon: <CloseIcon /> } };
export const Disabled: Story = { args: { disabled: true, label: 'Delete', icon: <TrashIcon /> } };
