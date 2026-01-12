"""Debug script to check if UCCL is available on Ray actors."""
import ray


@ray.remote(num_gpus=1)
class DebugActor:
    def check_uccl(self):
        import sys
        results = []

        # Check 1: Can we import uccl?
        try:
            from uccl import p2p
            results.append("✓ uccl.p2p imported successfully")
        except ImportError as e:
            results.append(f"✗ Failed to import uccl.p2p: {e}")
            return "\n".join(results)

        # Check 2: Can we import the transport manager?
        try:
            from ray.experimental.gpu_object_manager.util import get_tensor_transport_manager
            results.append("✓ get_tensor_transport_manager imported")
        except ImportError as e:
            results.append(f"✗ Failed to import get_tensor_transport_manager: {e}")
            return "\n".join(results)

        # Check 3: Can we get the UCCL transport manager?
        try:
            manager = get_tensor_transport_manager("UCCL")
            results.append(f"✓ Got UCCL transport manager: {type(manager)}")
        except Exception as e:
            results.append(f"✗ Failed to get UCCL transport manager: {e}")
            return "\n".join(results)

        # Check 4: Can we create an endpoint?
        try:
            endpoint = manager._get_uccl_endpoint()
            results.append(f"✓ Created UCCL endpoint: {endpoint}")
        except Exception as e:
            import traceback
            results.append(f"✗ Failed to create UCCL endpoint: {e}")
            results.append(f"  Traceback: {traceback.format_exc()}")
            return "\n".join(results)

        # Check 5: Can we get metadata?
        try:
            metadata = endpoint.get_metadata()
            ip, port, gpu = p2p.Endpoint.parse_metadata(metadata)
            results.append(f"✓ Endpoint metadata: IP={ip}, port={port}, GPU={gpu}")
        except Exception as e:
            results.append(f"✗ Failed to get endpoint metadata: {e}")

        return "\n".join(results)


def main():
    ray.init()

    actor = DebugActor.remote()
    result = ray.get(actor.check_uccl.remote())
    print("=== UCCL Debug Results ===")
    print(result)

    ray.shutdown()


if __name__ == "__main__":
    main()
