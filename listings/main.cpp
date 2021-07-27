#include <iostream>
using namespace std;

int main()
{
    cout << [](ostream& out) -> decltype(auto) { return out << "Hello world!\n"; } << endl;
    return 0;
}
