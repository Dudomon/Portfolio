import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

import { Button } from '../Button/Button';
import { TextField } from '../TextField/TextField';
import { Dialog, type DialogProps } from './Dialog';

function DialogDemo(props: Omit<DialogProps, 'open' | 'onClose'>) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button onClick={() => setOpen(true)}>Open dialog</Button>
      <Dialog
        {...props}
        open={open}
        onClose={() => setOpen(false)}
        footer={
          <>
            <Button onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="danger" onClick={() => setOpen(false)}>
              Reject document
            </Button>
          </>
        }
      />
    </>
  );
}

const meta = {
  title: 'UI/Dialog',
  component: DialogDemo,
  args: {
    title: 'Reject document',
    description: 'The supplier will need to resend a corrected invoice.',
    children: <TextField label="Reason" placeholder="e.g. Duplicate of NF-000122" />,
  },
} satisfies Meta<typeof DialogDemo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const AlertDialog: Story = {
  args: {
    role: 'alertdialog',
    title: 'Document changed',
    description: 'Someone else saved this document after you opened it.',
    children: undefined,
    size: 'sm',
  },
};

export const NotDismissible: Story = { args: { dismissible: false } };
