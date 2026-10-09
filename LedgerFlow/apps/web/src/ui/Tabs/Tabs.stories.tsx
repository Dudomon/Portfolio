import type { Meta, StoryObj } from '@storybook/react';

import { Tab, TabList, TabPanel, Tabs } from './Tabs';

const meta = {
  title: 'UI/Tabs',
  component: Tabs,
  args: { defaultValue: 'review', children: null },
  render: (args) => (
    <Tabs {...args}>
      <TabList aria-label="Document views">
        <Tab value="review">Review</Tab>
        <Tab value="audit">Audit trail</Tab>
        <Tab value="raw" disabled>
          Raw JSON
        </Tab>
      </TabList>
      <TabPanel value="review">
        <p style={{ padding: 16 }}>Editable extracted fields.</p>
      </TabPanel>
      <TabPanel value="audit">
        <p style={{ padding: 16 }}>Who changed what, and when.</p>
      </TabPanel>
      <TabPanel value="raw">
        <p style={{ padding: 16 }}>Unavailable.</p>
      </TabPanel>
    </Tabs>
  ),
} satisfies Meta<typeof Tabs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
export const SecondTabSelected: Story = { args: { defaultValue: 'audit' } };
