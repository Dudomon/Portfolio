import type { Meta, StoryObj } from '@storybook/react';

import { Badge, type BadgeTone } from './Badge';

const tones: BadgeTone[] = ['neutral', 'info', 'accent', 'success', 'warning', 'danger'];

const meta = {
  title: 'UI/Badge',
  component: Badge,
  args: { children: 'Needs review', tone: 'warning', dot: true },
  argTypes: { tone: { control: 'select', options: tones } },
} satisfies Meta<typeof Badge>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const AllTones: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {tones.map((tone) => (
        <Badge key={tone} tone={tone} dot>
          {tone}
        </Badge>
      ))}
    </div>
  ),
};

export const WithoutDot: Story = { args: { tone: 'neutral', dot: false, children: 'LLM' } };
