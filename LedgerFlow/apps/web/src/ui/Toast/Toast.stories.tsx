import type { Meta, StoryObj } from '@storybook/react';

import { Button } from '../Button/Button';
import { ToastProvider, useToast, type ToastTone } from './Toast';

function ToastDemo({
  tone,
  title,
  description,
}: {
  tone: ToastTone;
  title: string;
  description?: string;
}) {
  const toast = useToast();
  return (
    <Button onClick={() => toast.show({ tone, title, description })}>Show {tone} toast</Button>
  );
}

const meta = {
  title: 'UI/Toast',
  component: ToastDemo,
  decorators: [
    (Story) => (
      <ToastProvider>
        <Story />
      </ToastProvider>
    ),
  ],
  args: { tone: 'success', title: 'Document approved', description: 'invoice-0042.pdf' },
  argTypes: {
    tone: { control: 'inline-radio', options: ['info', 'success', 'warning', 'danger'] },
  },
} satisfies Meta<typeof ToastDemo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Success: Story = {};
export const Info: Story = { args: { tone: 'info', title: 'Upload started' } };
export const Warning: Story = { args: { tone: 'warning', title: 'Save your corrections first' } };
export const DangerIsSticky: Story = {
  args: { tone: 'danger', title: 'Upload failed', description: 'File exceeds 15 MB.' },
};
