import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';

import { FileDropzone } from './FileDropzone';

const meta = {
  title: 'UI/FileDropzone',
  component: FileDropzone,
  args: {
    accept: ['application/pdf', 'image/png', 'image/jpeg', '.xml'],
    maxSizeBytes: 15 * 1024 * 1024,
    label: 'Drop invoices here',
    hint: 'PDF, PNG, JPEG or NF-e XML',
    multiple: true,
    onFilesAccepted: fn(),
    onFilesRejected: fn(),
  },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 480 }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof FileDropzone>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
export const SingleFile: Story = { args: { multiple: false } };
export const Disabled: Story = { args: { disabled: true } };
