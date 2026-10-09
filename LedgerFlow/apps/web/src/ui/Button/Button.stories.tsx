import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';

import { CheckIcon } from '../icons';
import { Button } from './Button';

const meta = {
  title: 'UI/Button',
  component: Button,
  args: { children: 'Save corrections', onClick: fn() },
  argTypes: {
    variant: { control: 'inline-radio', options: ['primary', 'secondary', 'ghost', 'danger'] },
    size: { control: 'inline-radio', options: ['sm', 'md'] },
  },
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = { args: { variant: 'primary' } };
export const Secondary: Story = {};
export const Ghost: Story = { args: { variant: 'ghost' } };
export const Danger: Story = { args: { variant: 'danger', children: 'Reject' } };
export const Loading: Story = { args: { variant: 'primary', loading: true } };
export const Disabled: Story = { args: { disabled: true } };

export const WithIconAndShortcut: Story = {
  args: { variant: 'primary', children: 'Approve', startIcon: <CheckIcon />, shortcut: 'a' },
};

export const Sizes: Story = {
  render: (args) => (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <Button {...args} size="sm" />
      <Button {...args} size="md" />
    </div>
  ),
};
