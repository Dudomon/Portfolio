import type { Meta, StoryObj } from '@storybook/react';

import { ConfidenceMeter } from './ConfidenceMeter';

const meta = {
  title: 'UI/ConfidenceMeter',
  component: ConfidenceMeter,
  args: { value: 0.93 },
  argTypes: { value: { control: { type: 'range', min: 0, max: 1, step: 0.01 } } },
} satisfies Meta<typeof ConfidenceMeter>;

export default meta;
type Story = StoryObj<typeof meta>;

export const High: Story = {};
export const Borderline: Story = { args: { value: 0.72 } };
export const Low: Story = { args: { value: 0.41 } };
export const Unknown: Story = { args: { value: null } };
export const CustomThreshold: Story = { args: { value: 0.88, threshold: 0.95 } };
export const WithoutValue: Story = { args: { showValue: false } };
