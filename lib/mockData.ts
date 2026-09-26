import type { Drone, FleetEvent, Member } from "./types";

const mariaDrones: Drone[] = [
  { id: "Drone-1", status: "Idle", batteryPct: 80, currentTask: "Mission complete" },
  { id: "Drone-2", status: "Idle", batteryPct: 15, currentTask: "Mission complete" },
  { id: "Drone-3", status: "Idle", batteryPct: 88, currentTask: "Mission complete" },
  { id: "Drone-4", status: "Idle", batteryPct: 71, currentTask: "Mission complete" },
];

const mariaEvents: FleetEvent[] = [
  {
    timestamp: "10:01:58",
    droneId: "Drone-1",
    type: "capture",
    message: "Drone-1 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 98,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "10:01:59",
    droneId: "Drone-2",
    type: "capture",
    message: "Drone-2 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 97,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "10:02:00",
    droneId: "Drone-3",
    type: "capture",
    message: "Drone-3 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 99,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "10:02:01",
    droneId: "Drone-4",
    type: "capture",
    message: "Drone-4 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 96,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "10:02:14",
    droneId: "Drone-1",
    type: "capture",
    message: "Drone-1 capturing meter closeup",
    status: "Flying",
    batteryPct: 94,
    currentTask: "Capturing meter closeup",
  },
  {
    timestamp: "10:02:20",
    droneId: "Drone-3",
    type: "capture",
    message: "Drone-3 capturing left wall clearance",
    status: "Flying",
    batteryPct: 95,
    currentTask: "Capturing left wall clearance",
  },
  {
    timestamp: "10:02:35",
    droneId: "Drone-4",
    type: "capture",
    message: "Drone-4 capturing rear yard wall",
    status: "Flying",
    batteryPct: 92,
    currentTask: "Capturing rear yard wall",
  },
  {
    timestamp: "10:03:40",
    droneId: "Drone-2",
    type: "low_battery",
    message: "Drone-2 battery at 15%, returning to charge",
    status: "Charging",
    batteryPct: 15,
    currentTask: "Returning to charge",
  },
  {
    timestamp: "10:03:41",
    droneId: "Drone-2",
    type: "reassign",
    message: "Task 'capture panel label' reassigned Drone-2 -> Drone-3",
    targetDroneId: "Drone-3",
    targetStatus: "Flying",
    targetTask: "Capture panel label",
  },
  {
    timestamp: "10:04:10",
    droneId: "Drone-3",
    type: "capture",
    message: "Drone-3 capturing panel label",
    status: "Flying",
    batteryPct: 88,
    currentTask: "Capturing panel label",
  },
  {
    timestamp: "10:05:02",
    droneId: "Drone-1",
    type: "failure",
    message: "Drone-1 lost signal, task requeued",
    status: "Failed",
    batteryPct: 80,
    currentTask: "Signal lost",
  },
  {
    timestamp: "10:05:03",
    droneId: "Drone-1",
    type: "reassign",
    message: "Task 'capture garage sidewall' reassigned Drone-1 -> Drone-4",
    targetDroneId: "Drone-4",
    targetStatus: "Flying",
    targetTask: "Capture garage sidewall",
  },
  {
    timestamp: "10:05:40",
    droneId: "Drone-4",
    type: "capture",
    message: "Drone-4 capturing garage sidewall",
    status: "Flying",
    batteryPct: 71,
    currentTask: "Capturing garage sidewall",
  },
  {
    timestamp: "10:06:30",
    droneId: "Drone-1",
    type: "complete",
    message: "Capture complete: 12 photos, 0 tasks lost",
  },
];

const jamesDrones: Drone[] = [
  { id: "Drone-1", status: "Idle", batteryPct: 93, currentTask: "Mission complete" },
  { id: "Drone-2", status: "Idle", batteryPct: 94, currentTask: "Mission complete" },
  { id: "Drone-3", status: "Idle", batteryPct: 90, currentTask: "Mission complete" },
  { id: "Drone-4", status: "Idle", batteryPct: 95, currentTask: "Mission complete" },
];

const jamesEvents: FleetEvent[] = [
  {
    timestamp: "09:15:02",
    droneId: "Drone-1",
    type: "capture",
    message: "Drone-1 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 97,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "09:15:03",
    droneId: "Drone-2",
    type: "capture",
    message: "Drone-2 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 98,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "09:15:04",
    droneId: "Drone-3",
    type: "capture",
    message: "Drone-3 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 96,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "09:15:05",
    droneId: "Drone-4",
    type: "capture",
    message: "Drone-4 launched, heading to equipment wall",
    status: "Flying",
    batteryPct: 99,
    currentTask: "Heading to equipment wall",
  },
  {
    timestamp: "09:15:20",
    droneId: "Drone-1",
    type: "capture",
    message: "Drone-1 capturing meter closeup",
    status: "Flying",
    batteryPct: 93,
    currentTask: "Capturing meter closeup",
  },
  {
    timestamp: "09:15:45",
    droneId: "Drone-2",
    type: "capture",
    message: "Drone-2 capturing garage sidewall",
    status: "Flying",
    batteryPct: 94,
    currentTask: "Capturing garage sidewall",
  },
  {
    timestamp: "09:16:10",
    droneId: "Drone-3",
    type: "capture",
    message: "Drone-3 capturing rear yard wall",
    status: "Flying",
    batteryPct: 90,
    currentTask: "Capturing rear yard wall",
  },
  {
    timestamp: "09:16:40",
    droneId: "Drone-4",
    type: "capture",
    message: "Drone-4 capturing panel label",
    status: "Flying",
    batteryPct: 95,
    currentTask: "Capturing panel label",
  },
  {
    timestamp: "09:17:15",
    droneId: "Drone-1",
    type: "complete",
    message: "Capture complete: 8 photos, 0 tasks lost",
  },
];

export const mockMembers: Member[] = [
  {
    id: "mem-001",
    name: "Maria Chen",
    address: "1842 Oak Ridge Dr, Austin, TX 78745",
    email: "maria.chen@example.com",
    reportStatus: "Ready for Review",
    report: {
      drones: mariaDrones,
      events: mariaEvents,
      placementPhotos: [
        {
          id: "place-001-a",
          imageUrl: "/placeholder/wall-garage.svg",
          label: "Option A: garage sidewall",
          distanceToMeterFt: 18,
          clearancePass: true,
          placementBoxes: [
            { box: { x: 30, y: 44, width: 22, height: 30 }, label: "Battery" },
            { box: { x: 58, y: 40, width: 14, height: 18 }, label: "Disconnect" },
          ],
        },
        {
          id: "place-001-b",
          imageUrl: "/placeholder/wall-left.svg",
          label: "Option B: left of meter",
          distanceToMeterFt: 8,
          clearancePass: false,
          placementBoxes: [
            { box: { x: 18, y: 48, width: 24, height: 28 }, label: "Battery" },
            { box: { x: 50, y: 42, width: 13, height: 17 }, label: "Disconnect" },
          ],
        },
      ],
      blockerPhotos: [
        {
          id: "block-001-a",
          imageUrl: "/placeholder/wall-left.svg",
          blockers: [
            {
              id: "b-001-1",
              type: "Vegetation",
              severity: "Medium",
              description: "Bush within 3 ft of meter.",
              requiredFix: "Trim to 3 ft clearance.",
              box: { x: 28, y: 58, width: 22, height: 18 },
            },
            {
              id: "b-001-2",
              type: "Obstructed Meter",
              severity: "High",
              description: "Meter face partially obscured in capture.",
              requiredFix:
                "Clear all items in front of the meter and send an updated photo.",
              box: { x: 32, y: 22, width: 20, height: 24 },
            },
          ],
        },
        {
          id: "block-001-b",
          imageUrl: "/placeholder/wall-center.svg",
          blockers: [
            {
              id: "b-001-3",
              type: "AC Unit",
              severity: "Low",
              description: "AC condenser near proposed conduit path.",
              requiredFix:
                "Maintain 3 ft clearance around the condenser for conduit routing.",
              box: { x: 62, y: 52, width: 24, height: 22 },
            },
          ],
        },
      ],
    },
  },
  {
    id: "mem-002",
    name: "James Okonkwo",
    address: "903 Cedar Hollow Ln, Round Rock, TX 78664",
    email: "j.okonkwo@example.com",
    reportStatus: "Ready for Review",
    report: {
      drones: jamesDrones,
      events: jamesEvents,
      placementPhotos: [
        {
          id: "place-002-a",
          imageUrl: "/placeholder/wall-garage.svg",
          label: "Option A: garage sidewall",
          distanceToMeterFt: 20,
          clearancePass: true,
          placementBoxes: [
            { box: { x: 32, y: 46, width: 22, height: 28 }, label: "Battery" },
            { box: { x: 60, y: 42, width: 14, height: 18 }, label: "Disconnect" },
          ],
        },
      ],
      blockerPhotos: [],
    },
  },
  {
    id: "mem-003",
    name: "Patricia Reyes",
    address: "4410 Magnolia Bend, Pflugerville, TX 78660",
    email: "patricia.reyes@example.com",
    reportStatus: "Awaiting Drone Report",
    report: null,
  },
];

export function getMemberById(id: string): Member | undefined {
  return mockMembers.find((m) => m.id === id);
}

export function countMemberBlockers(member: Member): number {
  if (!member.report) return 0;
  return member.report.blockerPhotos.reduce(
    (sum, photo) => sum + photo.blockers.length,
    0,
  );
}
