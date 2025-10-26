# Check list

Remove the capturing part first

And try to load and run the model standalone

So that we can fix errors arising from model loading part

If model is loading, Then live packet capture can integrated later!

# Future proof:

## Thread1:
Capture sequence of packets from scapy and store in temporary buffer

## Thread2:
Fetch the packets from the buffer, then convert them into flow
with required features



