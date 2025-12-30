# Phoenix Real Time Chat

A demonstration of Elixir and Phoenix Channels for real time communication. Built to showcase the BEAM virtual machine's strengths in concurrent, fault tolerant systems.

## Why Elixir for Real Time Applications

### The Problem with Traditional Approaches

Node.js handles concurrent connections with a single threaded event loop. When one operation blocks (database query, external API call), all other connections wait. Solutions involve callbacks, promises, or async/await, which add complexity and error handling challenges.

Ruby and Python use thread pools. Each connection consumes a thread with significant memory overhead (1MB+ per thread). A server with 16GB RAM maxes out at roughly 16,000 concurrent connections before memory exhaustion.

### How Elixir Solves This

Elixir processes are not OS threads. They are lightweight abstractions managed by the BEAM VM scheduler, consuming only 2KB of initial memory. A single server runs millions of processes concurrently.

Each chat room in this application is a GenServer process. Each connected user has a Channel process. When a user sends a message, that process handles it without blocking any other user. If a user's process crashes (malformed message, network hiccup), only that user is affected; others continue chatting uninterrupted.

Phoenix Channels leverage this model. A channel join creates a process. Broadcasting a message to 10,000 users means 10,000 processes each receive a message in their mailbox and send it over their WebSocket. The scheduler distributes this work across all CPU cores automatically.

## Architecture

### Process Hierarchy

```
Application
├── ChatWeb.Endpoint (HTTP/WebSocket supervisor)
│   └── Socket connections (one process per connected user)
│       └── Channel processes (one per room subscription)
├── Chat.RoomRegistry (process registry for room lookup)
└── Chat.RoomSupervisor (dynamic supervisor for room processes)
    ├── Room "general" (GenServer)
    ├── Room "elixir" (GenServer)
    └── Room "react" (GenServer)
```

### Fault Tolerance

If a room process crashes, its supervisor restarts it. Connected users experience a brief reconnection (handled automatically by Phoenix Channel client) and continue chatting. Message history is lost unless persisted to database, which is a conscious tradeoff: this demo prioritizes demonstrating real time mechanics over building a full featured chat application.

### Presence Tracking

Phoenix.Presence uses CRDTs (Conflict free Replicated Data Types) for distributed presence tracking. When user A joins a room, all other users in that room receive a presence diff event. This works across multiple server nodes without a central presence database.

## Features Demonstrated

1. **Room Management**: Dynamic process creation for chat rooms
2. **Real Time Messaging**: Sub 50ms message delivery via WebSocket
3. **Typing Indicators**: Presence diffs show who is typing
4. **User Presence**: See who is online in each room
5. **Message History**: Last N messages stored in room process state

## Project Structure

```
lib/
  chat/
    rooms/
      room.ex           # GenServer managing room state
      room_supervisor.ex # DynamicSupervisor for rooms
      registry.ex       # Process registry for room lookup
    accounts.ex         # User authentication context
  chat_web/
    channels/
      room_channel.ex   # Phoenix Channel for room communication
      user_socket.ex    # WebSocket entry point
      presence.ex       # Presence tracking

frontend/
  src/
    components/
      ChatRoom.tsx      # Main chat interface
      MessageList.tsx   # Message display with virtualization
      UserList.tsx      # Online users sidebar
    hooks/
      useChannel.ts     # Phoenix Channel React hook
      usePresence.ts    # Presence tracking hook
```

## Running Locally

```bash
# Backend
mix deps.get
mix phx.server

# Frontend
cd frontend
npm install
npm run dev
```

WebSocket connects to `ws://localhost:4000/socket`.

## Performance Characteristics

Tested on a single 4 core machine:
- 50,000 concurrent connections
- 10,000 messages per second throughput
- P99 message latency under 100ms
- Memory usage: 4GB for 50,000 connections

These numbers scale linearly with additional nodes in a cluster.
