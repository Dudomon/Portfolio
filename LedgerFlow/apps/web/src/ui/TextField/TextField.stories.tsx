import type { Meta, StoryObj } from '@storybook/react';

import { ConfidenceMeter } from '../ConfidenceMeter/ConfidenceMeter';
import { TextField } from './TextField';

const meta = {
  title: 'UI/TextField',
  component: TextField,
  args: { label: 'Invoice number', defaultValue: 'NF-000123' },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 360 }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof TextField>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
export const WithHint: Story = { args: { hint: 'As printed on the document.' } };
export const WithError: Story = {
  args: {
    label: 'Supplier tax ID',
    defaultValue: '1234',
    error: 'Enter a 14-digit CNPJ or a VAT id.',
  },
};
export const WithLabelAccessory: Story = {
  args: { labelAccessory: <ConfidenceMeter value={0.62} label="Invoice number confidence" /> },
};
export const Small: Story = { args: { size: 'sm' } };
export const Disabled: Story = { args: { disabled: true } };
export const HiddenLabel: Story = { args: { hideLabel: true, placeholder: 'Search documents' } };
